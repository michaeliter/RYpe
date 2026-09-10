//! Parquet input reading with background prefetching.

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, LargeStringArray, RecordBatch, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::{FirstErrorCapture, QueryRecord};

use super::fastx_io::OwnedFastxRecord;

/// Bounded semaphore gating how many row groups may be claimed-but-not-yet-
/// emitted at once in `reader_thread_parallel`'s bounded work queue. Without
/// this, workers can race arbitrarily far ahead of a straggler row group
/// (or a slow consumer), buffering unboundedly decoded data in the reorder
/// buffer -- this caps that to `capacity` outstanding row groups, matching
/// the old chunked design's memory bound while still letting workers keep
/// claiming new work as soon as the *oldest* unemitted one clears, rather
/// than waiting for an entire fixed-size chunk to finish.
struct RowGroupWindow {
    state: Mutex<RowGroupWindowState>,
    cond: Condvar,
}

struct RowGroupWindowState {
    available: usize,
    aborted: bool,
}

impl RowGroupWindow {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(RowGroupWindowState {
                available: capacity,
                aborted: false,
            }),
            cond: Condvar::new(),
        }
    }

    /// Blocks until a permit is available or the window is aborted.
    /// Returns `false` (no permit acquired) if aborted.
    fn acquire(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        while state.available == 0 && !state.aborted {
            state = self.cond.wait(state).unwrap();
        }
        if state.aborted {
            return false;
        }
        state.available -= 1;
        true
    }

    /// Returns a permit, waking one waiter (if any).
    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.available += 1;
        self.cond.notify_one();
    }

    /// Wakes every current and future waiter without granting a permit --
    /// used when the emitter gives up early (error, or the consumer
    /// dropped) so workers blocked on `acquire()` don't hang forever
    /// waiting for a permit that will never come.
    fn abort(&self) {
        let mut state = self.state.lock().unwrap();
        state.aborted = true;
        self.cond.notify_all();
    }
}

/// Check if a file path indicates Parquet input.
pub fn is_parquet_input(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("parquet"))
        .unwrap_or(false)
}

/// Reader for Parquet input files with read_id, sequence1, and optional sequence2 columns.
#[allow(dead_code)]
pub struct ParquetInputReader {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    is_paired: bool,
    current_batch: Option<RecordBatch>,
    current_idx: usize,
    global_record_id: i64,
}

#[allow(dead_code)]
impl ParquetInputReader {
    /// Check if a data type is a valid string type (Utf8 or LargeUtf8).
    fn is_string_type(dt: &DataType) -> bool {
        matches!(dt, DataType::Utf8 | DataType::LargeUtf8)
    }

    /// Open a Parquet file and validate schema.
    ///
    /// Required columns: read_id (string), sequence1 (string)
    /// Optional column: sequence2 (string) - if first row is non-null, data is paired
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file cannot be opened
    /// - Required columns are missing
    /// - Columns have incorrect types (must be string types)
    pub fn new(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Failed to open Parquet file: {:?}", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader")?;

        let schema = builder.schema();

        // Validate required columns exist and have correct types
        let read_id_field = schema
            .fields()
            .iter()
            .find(|f| f.name() == "read_id")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'read_id'"))?;

        if !Self::is_string_type(read_id_field.data_type()) {
            return Err(anyhow!(
                "Column 'read_id' must be string type (Utf8 or LargeUtf8), got {:?}",
                read_id_field.data_type()
            ));
        }

        let sequence1_field = schema
            .fields()
            .iter()
            .find(|f| f.name() == "sequence1")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'sequence1'"))?;

        if !Self::is_string_type(sequence1_field.data_type()) {
            return Err(anyhow!(
                "Column 'sequence1' must be string type (Utf8 or LargeUtf8), got {:?}",
                sequence1_field.data_type()
            ));
        }

        // Check optional sequence2 column
        let has_sequence2 =
            if let Some(field) = schema.fields().iter().find(|f| f.name() == "sequence2") {
                if !Self::is_string_type(field.data_type()) {
                    return Err(anyhow!(
                        "Column 'sequence2' must be string type (Utf8 or LargeUtf8), got {:?}",
                        field.data_type()
                    ));
                }
                true
            } else {
                false
            };

        let mut reader = builder.build().context("Failed to build Parquet reader")?;

        // Read first batch to detect if paired-end (sequence2 non-null)
        let first_batch = reader.next();
        let (is_paired, current_batch) = match first_batch {
            Some(Ok(batch)) => {
                let is_paired = if has_sequence2 {
                    // Check if first value in sequence2 is non-null
                    if let Some(col) = batch.column_by_name("sequence2") {
                        col.null_count() < col.len()
                    } else {
                        false
                    }
                } else {
                    false
                };
                (is_paired, Some(batch))
            }
            Some(Err(e)) => return Err(anyhow!("Error reading first Parquet batch: {}", e)),
            None => (false, None), // Empty file
        };

        // Log paired-end detection result
        log::info!(
            "Parquet input '{}': detected as {} data",
            path.display(),
            if is_paired {
                "paired-end"
            } else {
                "single-end"
            }
        );

        Ok(Self {
            reader,
            is_paired,
            current_batch,
            current_idx: 0,
            global_record_id: 0,
        })
    }

    /// Returns whether the input is paired-end.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.is_paired
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some((records, headers)))` for each batch,
    /// `Ok(None)` when all records have been read.
    pub fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Result<Option<(Vec<OwnedFastxRecord>, Vec<String>)>> {
        let mut records = Vec::with_capacity(batch_size);
        let mut headers = Vec::with_capacity(batch_size);

        while records.len() < batch_size {
            // Get next record from current batch
            if let Some(ref batch) = self.current_batch {
                if self.current_idx < batch.num_rows() {
                    // Extract columns
                    let read_id_col = batch
                        .column_by_name("read_id")
                        .ok_or_else(|| anyhow!("Missing read_id column"))?;
                    let sequence1_col = batch
                        .column_by_name("sequence1")
                        .ok_or_else(|| anyhow!("Missing sequence1 column"))?;

                    let read_id_arr = read_id_col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("read_id column is not a string array"))?;
                    let sequence1_arr = sequence1_col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("sequence1 column is not a string array"))?;

                    let sequence2_arr = if self.is_paired {
                        batch
                            .column_by_name("sequence2")
                            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    } else {
                        None
                    };

                    let idx = self.current_idx;
                    let read_id = read_id_arr.value(idx).to_string();
                    let sequence1 = sequence1_arr.value(idx).as_bytes().to_vec();
                    let sequence2 = sequence2_arr
                        .filter(|arr| !arr.is_null(idx))
                        .map(|arr| arr.value(idx).as_bytes().to_vec());

                    // Use batch-local index for query_id (will be adjusted in output)
                    records.push(OwnedFastxRecord::new(
                        records.len() as i64,
                        sequence1,
                        None, // qual1 - not available in Parquet input
                        sequence2,
                        None, // qual2 - not available in Parquet input
                    ));
                    headers.push(read_id);
                    self.global_record_id += 1;
                    self.current_idx += 1;
                    continue;
                }
            }

            // Need to load next batch
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.current_batch = Some(batch);
                    self.current_idx = 0;
                }
                Some(Err(e)) => return Err(anyhow!("Error reading Parquet batch: {}", e)),
                None => {
                    // No more batches
                    break;
                }
            }
        }

        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some((records, headers)))
        }
    }
}

// ============================================================================
// Prefetching Parquet Reader
// ============================================================================

/// A batch of data from the Parquet reader, either zero-copy Arrow or owned records.
///
/// The reader thread produces `Arrow` batches by default (zero-copy path).
/// When trim/filter options are active (Cycle 7+), the reader thread converts
/// batches to `Owned` records before sending, so the large Arrow buffers are
/// freed immediately in the reader thread.
pub enum ParquetBatch {
    /// Zero-copy Arrow batch with pre-extracted headers.
    Arrow(RecordBatch, Vec<String>),
    /// Owned records (trimmed/filtered) with headers.
    Owned(Vec<OwnedFastxRecord>, Vec<String>),
}

impl ParquetBatch {
    /// Unwrap as Arrow variant, panicking if Owned.
    ///
    /// Use this in code paths that know trimming/filtering is not active.
    pub fn into_arrow(self) -> (RecordBatch, Vec<String>) {
        match self {
            ParquetBatch::Arrow(batch, headers) => (batch, headers),
            ParquetBatch::Owned(..) => {
                panic!("Expected ParquetBatch::Arrow but got Owned")
            }
        }
    }

    /// Unwrap as Owned variant, panicking if Arrow.
    ///
    /// Use this in code paths that know trimming/filtering is active.
    #[allow(dead_code)]
    pub fn into_owned(self) -> (Vec<OwnedFastxRecord>, Vec<String>) {
        match self {
            ParquetBatch::Owned(records, headers) => (records, headers),
            ParquetBatch::Arrow(..) => {
                panic!("Expected ParquetBatch::Owned but got Arrow")
            }
        }
    }
}

/// Type alias for Parquet batch data sent through the prefetch channel.
type ParquetBatchResult = Result<Option<ParquetBatch>>;

/// Default timeout for waiting on Parquet prefetch batches (5 minutes).
const DEFAULT_PARQUET_PREFETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Prefetching Parquet reader with background I/O for overlapping I/O and computation.
///
/// This reader spawns a background thread that reads RecordBatch objects from Parquet
/// while the main thread processes the current batch. Unlike `ParquetInputReader` which
/// copies sequences record-by-record, this reader enables **zero-copy** sequence access
/// by keeping the RecordBatch alive during classification.
///
/// # Zero-Copy Design
///
/// The background thread reads complete RecordBatch objects and extracts only headers
/// (read_ids). The main thread then uses `batch_to_records_parquet()` to get zero-copy
/// references into the Arrow buffer memory.
///
/// # Usage
///
/// ```ignore
/// let mut reader = PrefetchingParquetReader::new(&path)?;
/// while let Some((batch, headers)) = reader.next_batch()? {
///     // Zero-copy conversion - references point into batch's Arrow buffers
///     let records: Vec<QueryRecord> = batch_to_records_parquet(&batch, headers.len())?;
///     let results = classify_batch(..., &records, ...);
///     // batch must stay alive until classification completes
/// }
/// ```
pub struct PrefetchingParquetReader {
    receiver: Receiver<ParquetBatchResult>,
    prefetch_thread: Option<JoinHandle<()>>,
    error_capture: Arc<FirstErrorCapture>,
    is_paired: bool,
    timeout: Duration,
}

impl PrefetchingParquetReader {
    /// Create a new prefetching Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of records per batch (controls Parquet reader batch size)
    ///
    /// # Returns
    /// A reader that prefetches RecordBatch objects in a background thread.
    #[allow(dead_code)]
    pub fn new(path: &Path, batch_size: usize) -> Result<Self> {
        Self::with_parallel_row_groups(path, batch_size, None, None, None)
    }

    /// Create a new prefetching Parquet reader with optional parallel row group processing.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of records per batch (controls Parquet reader batch size)
    /// * `parallel_row_groups` - Optional number of row groups to read concurrently.
    ///   If `Some(n)`, uses a bounded work queue of up to n concurrent readers.
    ///   If `None`, uses sequential reading (original behavior).
    ///
    /// # Returns
    /// A reader that prefetches RecordBatch objects in a background thread.
    ///
    /// # Parallel Reading
    ///
    /// When `parallel_row_groups` is `Some(n)`:
    /// - Up to n row groups are read concurrently at any time; each worker claims the
    ///   next unclaimed row group as soon as it finishes its current one, rather than
    ///   waiting for a fixed-size batch of n to fully complete
    /// - Each parallel task opens its own file handle
    /// - A reorder buffer re-serializes results to maintain row-group order, bounded to
    ///   at most n claimed-but-not-yet-emitted row groups at once (see `RowGroupWindow`)
    ///   so a slow row group can't let unbounded decoded data accumulate in memory
    /// - Most effective when decompression is CPU-bound (not I/O-bound)
    ///
    /// Recommended values:
    /// - `Some(4)` - Good balance for most SSDs (default when enabled)
    /// - `Some(2)` - More conservative, lower memory usage
    /// - `None` - Sequential reading (original behavior)
    pub fn with_parallel_row_groups(
        path: &Path,
        batch_size: usize,
        parallel_row_groups: Option<usize>,
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
    ) -> Result<Self> {
        // Clone path for the background thread
        let path = path.to_path_buf();

        // Thread-safe error capture (only stores first error)
        let error_capture = Arc::new(FirstErrorCapture::new());
        let thread_error = Arc::clone(&error_capture);

        // Use sync_channel with buffer of 4 for prefetching:
        // - Larger buffer allows more read-ahead when classification is slower than decompression
        // - Trade-off: more memory usage for buffered RecordBatches
        let (sender, receiver): (SyncSender<ParquetBatchResult>, Receiver<ParquetBatchResult>) =
            mpsc::sync_channel(4);

        // Read parquet schema to validate and get column indices
        let file = File::open(&path)
            .with_context(|| format!("Failed to open Parquet file: {:?}", path))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader")?;
        let schema = builder.schema().clone();

        // Validate schema and get column indices for projection
        let (col_indices, has_sequence2) = Self::validate_and_get_projection(&schema)?;

        // Spawn background thread - either parallel or sequential based on option
        // Note: Some(0) is treated as disabled (sequential mode) - users who want
        // default parallelism should use Some(DEFAULT_PARALLEL_ROW_GROUPS) explicitly
        let prefetch_thread = if let Some(parallel_rg) = parallel_row_groups.filter(|&n| n > 0) {
            log::info!(
                "Using parallel Parquet row group reading (parallelism={})",
                parallel_rg
            );
            thread::spawn(move || {
                Self::reader_thread_parallel(
                    path,
                    batch_size,
                    col_indices,
                    parallel_rg,
                    trim_to,
                    minimum_length,
                    sender,
                    thread_error,
                );
            })
        } else {
            thread::spawn(move || {
                Self::reader_thread(
                    path,
                    batch_size,
                    col_indices,
                    trim_to,
                    minimum_length,
                    sender,
                    thread_error,
                );
            })
        };

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            error_capture,
            is_paired: has_sequence2,
            timeout: DEFAULT_PARQUET_PREFETCH_TIMEOUT,
        })
    }

    /// Validate the Parquet schema and return column indices for projection.
    ///
    /// Returns (column_indices, has_sequence2) where column_indices contains
    /// the indices of read_id, sequence1, and optionally sequence2.
    fn validate_and_get_projection(
        schema: &arrow::datatypes::Schema,
    ) -> Result<(Vec<usize>, bool)> {
        fn is_string_type(dt: &DataType) -> bool {
            matches!(dt, DataType::Utf8 | DataType::LargeUtf8)
        }

        // Find read_id column
        let read_id_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "read_id")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'read_id'"))?;

        if !is_string_type(schema.field(read_id_idx).data_type()) {
            return Err(anyhow!(
                "Column 'read_id' must be string type (Utf8 or LargeUtf8), got {:?}",
                schema.field(read_id_idx).data_type()
            ));
        }

        // Find sequence1 column
        let sequence1_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "sequence1")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'sequence1'"))?;

        if !is_string_type(schema.field(sequence1_idx).data_type()) {
            return Err(anyhow!(
                "Column 'sequence1' must be string type (Utf8 or LargeUtf8), got {:?}",
                schema.field(sequence1_idx).data_type()
            ));
        }

        // Find optional sequence2 column
        let sequence2_idx = schema.fields().iter().position(|f| f.name() == "sequence2");

        if let Some(idx) = sequence2_idx {
            if !is_string_type(schema.field(idx).data_type()) {
                return Err(anyhow!(
                    "Column 'sequence2' must be string type (Utf8 or LargeUtf8), got {:?}",
                    schema.field(idx).data_type()
                ));
            }
        }

        // Build projection mask - only read the columns we need
        let mut col_indices = vec![read_id_idx, sequence1_idx];
        let has_sequence2 = sequence2_idx.is_some();
        if let Some(idx) = sequence2_idx {
            col_indices.push(idx);
        }

        Ok((col_indices, has_sequence2))
    }

    /// Background thread function that reads RecordBatches and sends them through the channel.
    ///
    /// Note: We do NOT use the user's batch_size here because Parquet's byte array decoder
    /// can overflow with very large batches (>200K rows with long sequences). Instead, we
    /// let the Parquet reader use natural row-group-based batching, which is safe.
    /// The main thread can accumulate multiple batches if larger batches are needed.
    fn reader_thread(
        path: PathBuf,
        _batch_size: usize, // Ignored - use natural Parquet batching to avoid overflow
        col_indices: Vec<usize>,
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
        sender: SyncSender<ParquetBatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
        let needs_trim_filter = trim_to.is_some() || minimum_length.is_some();
        // Helper macro to send error and store if send fails
        macro_rules! send_error {
            ($msg:expr) => {{
                let err_msg = $msg;
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    error_capture.store_msg(&err_msg);
                }
                return;
            }};
        }

        // Open file and create reader in background thread
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                send_error!(format!("Failed to open Parquet file: {}", e));
            }
        };

        let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(e) => {
                send_error!(format!("Failed to create Parquet reader: {}", e));
            }
        };

        // Build projection mask from column indices
        let projection =
            parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), col_indices);

        // Use column projection but let Parquet use natural row-group-based batching.
        // Do NOT use .with_batch_size() with large values as it can cause
        // "index overflow decoding byte array" errors with string columns.
        let reader = match builder.with_projection(projection).build() {
            Ok(r) => r,
            Err(e) => {
                send_error!(format!("Failed to build Parquet reader: {}", e));
            }
        };

        for batch_result in reader {
            let batch = match batch_result {
                Ok(b) => b,
                Err(e) => {
                    send_error!(format!("Error reading Parquet batch: {}", e));
                }
            };

            // Extract headers (read_ids) - these are small string copies
            let headers = match Self::extract_headers(&batch) {
                Ok(h) => h,
                Err(e) => {
                    send_error!(format!("Error extracting headers: {}", e));
                }
            };

            // If trim/filter active, convert to Owned in this thread so the channel
            // holds trimmed data (fixing the OOM bug with large --trim-to values).
            // Each batch uses id_offset=0; ID remapping happens during accumulation.
            let parquet_batch = if needs_trim_filter {
                match batch_to_owned_records_trimmed(&batch, &headers, trim_to, minimum_length, 0) {
                    Ok((records, filtered_headers)) => {
                        ParquetBatch::Owned(records, filtered_headers)
                    }
                    Err(e) => {
                        send_error!(format!("Error trimming batch: {}", e));
                    }
                }
            } else {
                ParquetBatch::Arrow(batch, headers)
            };

            if sender.send(Ok(Some(parquet_batch))).is_err() {
                // Receiver was dropped - exit cleanly
                return;
            }
        }

        // Send None to signal completion
        let _ = sender.send(Ok(None));
    }

    /// Background thread function that reads row groups concurrently via a
    /// bounded work queue and sends batches downstream in order.
    ///
    /// Up to `parallel_rg` worker tasks continuously claim the next unclaimed
    /// row group from a shared cursor (rather than processing fixed-size
    /// chunks), so a slow row group blocks only the emission of results that
    /// come after it in order -- not the start of unrelated later reads. A
    /// reorder buffer, bounded to `parallel_rg` claimed-but-not-yet-emitted
    /// row groups via `RowGroupWindow`, re-serializes results before
    /// forwarding them to `sender`.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `_batch_size` - Ignored (use natural Parquet batching to avoid overflow)
    /// * `col_indices` - Column indices for projection
    /// * `parallel_rg` - Max row groups read concurrently / buffered awaiting emission
    /// * `sender` - Channel sender for batch results
    /// * `error_capture` - Thread-safe error capture for errors during send failures
    #[allow(clippy::too_many_arguments)]
    fn reader_thread_parallel(
        path: PathBuf,
        _batch_size: usize, // Ignored - use natural Parquet batching to avoid overflow
        col_indices: Vec<usize>,
        parallel_rg: usize,
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
        sender: SyncSender<ParquetBatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
        let needs_trim_filter = trim_to.is_some() || minimum_length.is_some();
        // Helper macro to send error and store if send fails
        macro_rules! send_error {
            ($msg:expr) => {{
                let err_msg = $msg;
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    error_capture.store_msg(&err_msg);
                }
                return;
            }};
        }

        // Load metadata to get the number of row groups
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                send_error!(format!("Failed to open Parquet file: {}", e));
            }
        };

        let initial_metadata = match ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())
        {
            Ok(m) => m,
            Err(e) => {
                send_error!(format!("Failed to load Parquet metadata: {}", e));
            }
        };

        let num_row_groups = initial_metadata.metadata().num_row_groups();
        if num_row_groups == 0 {
            let _ = sender.send(Ok(None));
            return;
        }

        // Drop the initial file handle - each parallel task will open its own
        drop(file);

        log::debug!(
            "Parallel Parquet reader: {} row groups, parallelism={}",
            num_row_groups,
            parallel_rg
        );

        // Wrap col_indices in Arc for sharing across threads
        let col_indices = Arc::new(col_indices);
        let path = Arc::new(path);

        // Bounded work queue: up to `parallel_rg` workers continuously claim
        // the next unclaimed row group from a shared cursor, instead of the
        // old step_by(parallel_rg) chunking, where a slow row group blocked
        // both the rest of its chunk *and* the start of the next chunk. A
        // small reorder buffer re-serializes results before forwarding to
        // `sender`, since downstream consumers require row-group order.
        // `window` bounds how many row groups can be claimed-but-not-yet-
        // emitted at once (see `RowGroupWindow`), so a straggler can't let
        // decoded data pile up in `pending` without limit.
        //
        // Workers are plain OS threads, deliberately NOT rayon tasks: each
        // one runs a `loop` that never returns control until all row groups
        // are claimed, so scheduling them on rayon's shared global pool
        // would let `parallel_rg` of them permanently occupy pool workers
        // for the whole read. When `parallel_rg` is close to or exceeds
        // `rayon::current_num_threads()`, that starves every *other* rayon
        // caller sharing the pool -- including `process_batch()`'s own
        // `par_iter()` extraction on the consumer thread, which then blocks
        // forever waiting for a worker rayon will never free up (verified:
        // this reproduced as a deterministic hang via the real CLI with
        // `--parallel-input-rg` set to the machine's full thread count).
        let next_rg = Arc::new(AtomicUsize::new(0));
        let window = Arc::new(RowGroupWindow::new(parallel_rg.min(num_row_groups).max(1)));
        let (result_tx, result_rx) = mpsc::channel::<(usize, Result<Vec<ParquetBatch>, String>)>();

        let mut worker_handles = Vec::with_capacity(parallel_rg.min(num_row_groups));
        for _ in 0..parallel_rg.min(num_row_groups) {
            let next_rg = Arc::clone(&next_rg);
            let window = Arc::clone(&window);
            let result_tx = result_tx.clone();
            let path = Arc::clone(&path);
            let col_indices = Arc::clone(&col_indices);
            worker_handles.push(thread::spawn(move || loop {
                // Acquire a permit *before* claiming an index. If this were
                // reversed (claim, then acquire), a worker that claims a low
                // index (possibly the one the emitter is currently waiting
                // on) could be preempted between the two calls, letting
                // workers that claimed *later* indices win every remaining
                // permit first -- starving the low index's worker of a
                // permit forever, since none can be released until that
                // exact index is emitted. Acquiring first guarantees the
                // first `capacity` indices are always granted a permit
                // up front, with no ordering gap in between (verified: this
                // reproduced as a real, scheduling-dependent deadlock with
                // the old order, independent of rayon).
                if !window.acquire() {
                    break; // emitter aborted; no more permits will ever be granted
                }
                let rg_idx = next_rg.fetch_add(1, Ordering::Relaxed);
                if rg_idx >= num_row_groups {
                    window.release(); // no work left -- give back the unused permit
                    break;
                }
                let result = Self::read_row_group(
                    &path,
                    rg_idx,
                    &col_indices,
                    trim_to,
                    minimum_length,
                    needs_trim_filter,
                );
                if result_tx.send((rg_idx, result)).is_err() {
                    window.abort(); // emitter gave up -- unblock any siblings waiting on a permit
                    break;
                }
            }));
        }
        // Drop this thread's own clone so `result_rx.recv()` below sees the
        // channel close once every worker's clone is also dropped.
        drop(result_tx);

        // Emitter: runs on this thread, overlapping with the still-running
        // workers rather than waiting for them all to finish first,
        // reordering results as they arrive and forwarding them to `sender`
        // in row-group order. Every exit path falls through to joining the
        // workers below instead of returning early, so none are left
        // detached and still running past this function's return.
        let mut pending: HashMap<usize, Vec<ParquetBatch>> = HashMap::new();
        let mut next_to_emit = 0usize;
        while next_to_emit < num_row_groups {
            match result_rx.recv() {
                Ok((rg_idx, Ok(batches))) => {
                    pending.insert(rg_idx, batches);
                    while let Some(batches) = pending.remove(&next_to_emit) {
                        for parquet_batch in batches {
                            if sender.send(Ok(Some(parquet_batch))).is_err() {
                                window.abort();
                                for handle in worker_handles {
                                    let _ = handle.join();
                                }
                                return; // receiver dropped - exit cleanly
                            }
                        }
                        next_to_emit += 1;
                        window.release();
                    }
                }
                Ok((rg_idx, Err(e))) => {
                    error_capture.store_msg(format!("Error reading row group {}: {}", rg_idx, e));
                    window.abort();
                    for handle in worker_handles {
                        let _ = handle.join();
                    }
                    return;
                }
                Err(_) => {
                    // Every worker exited without completing every row
                    // group. Don't stamp a message here: if a worker
                    // panicked, joining below re-raises that panic on this
                    // thread, and the real panic message should win over a
                    // placeholder -- `next_batch()`'s disconnected-error
                    // handling already falls back to a clear message
                    // ("Prefetch thread panicked"/"exited unexpectedly")
                    // if `error_capture` is empty.
                    window.abort();
                    for handle in worker_handles {
                        let _ = handle.join();
                    }
                    return;
                }
            }
        }

        for handle in worker_handles {
            let _ = handle.join();
        }

        // Send None to signal completion, unless a row group failed above --
        // on error, falling through here (without sending) drops `sender`,
        // so `next_batch()` sees a closed channel and reports the message
        // stashed in `error_capture`.
        if error_capture.get().is_none() {
            let _ = sender.send(Ok(None));
        }
    }

    /// Read one row group's batches, applying trim/filter if active.
    ///
    /// Each call opens its own file handle and loads fresh metadata, which
    /// is more robust than sharing metadata/handles across threads.
    fn read_row_group(
        path: &Path,
        rg_idx: usize,
        col_indices: &[usize],
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
        needs_trim_filter: bool,
    ) -> Result<Vec<ParquetBatch>, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file for RG {}: {}", rg_idx, e))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("Failed to create reader for RG {}: {}", rg_idx, e))?;

        // Build projection mask for this reader
        let projection = parquet::arrow::ProjectionMask::roots(
            builder.parquet_schema(),
            col_indices.iter().copied(),
        );

        // Use row group selection and projection, but let Parquet use natural batching.
        // Do NOT use .with_batch_size() with large values to avoid offset overflow.
        let reader = builder
            .with_row_groups(vec![rg_idx])
            .with_projection(projection)
            .build()
            .map_err(|e| format!("Failed to build reader for RG {}: {}", rg_idx, e))?;

        // Collect all batches from this row group
        let mut batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| format!("Error reading batch from RG {}: {}", rg_idx, e))?;

            let headers = Self::extract_headers(&batch)
                .map_err(|e| format!("Error extracting headers from RG {}: {}", rg_idx, e))?;

            // If trim/filter active, convert to Owned in this thread.
            // Each batch uses id_offset=0; ID remapping happens during accumulation.
            if needs_trim_filter {
                let (records, filtered_headers) =
                    batch_to_owned_records_trimmed(&batch, &headers, trim_to, minimum_length, 0)
                        .map_err(|e| format!("Error trimming batch from RG {}: {}", rg_idx, e))?;
                batches.push(ParquetBatch::Owned(records, filtered_headers));
            } else {
                batches.push(ParquetBatch::Arrow(batch, headers));
            }
        }

        Ok(batches)
    }

    /// Extract headers (read_ids) from a RecordBatch.
    fn extract_headers(batch: &RecordBatch) -> Result<Vec<String>> {
        let col = batch
            .column_by_name("read_id")
            .ok_or_else(|| anyhow!("Missing read_id column"))?;

        // Try StringArray first, then LargeStringArray
        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            let mut headers = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                headers.push(arr.value(i).to_string());
            }
            return Ok(headers);
        }

        if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
            let mut headers = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                headers.push(arr.value(i).to_string());
            }
            return Ok(headers);
        }

        Err(anyhow!(
            "read_id column is not a string type: {:?}",
            col.data_type()
        ))
    }

    /// Returns whether the input is paired-end.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.is_paired
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some(ParquetBatch))` for each batch,
    /// `Ok(None)` when all records have been read,
    /// or `Err` if an error occurred during reading.
    ///
    /// The variant depends on reader configuration:
    /// - `ParquetBatch::Arrow` — zero-copy path (no trim/filter active)
    /// - `ParquetBatch::Owned` — owned records (trim/filter active, Cycle 7+)
    pub fn next_batch(&mut self) -> Result<Option<ParquetBatch>> {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                Err(anyhow!(
                    "Timeout waiting for next batch ({}s) - reader thread may be stalled",
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                if let Some(handle) = self.prefetch_thread.take() {
                    match handle.join() {
                        Ok(()) => Err(anyhow!("Prefetch thread exited unexpectedly")),
                        Err(_) => Err(anyhow!("Prefetch thread panicked")),
                    }
                } else {
                    Err(anyhow!("Prefetch channel closed"))
                }
            }
        }
    }

    /// Finish and wait for the prefetch thread to complete.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(handle) = self.prefetch_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow!("Prefetch thread panicked"))?;
        }
        Ok(())
    }
}

// ============================================================================
// Zero-Copy Batch Conversion
// ============================================================================

/// Enum for uniform access to String and LargeString sequence columns.
enum SequenceColumnRef<'a> {
    String(&'a StringArray),
    LargeString(&'a LargeStringArray),
}

impl<'a> SequenceColumnRef<'a> {
    #[inline]
    fn value(&self, i: usize) -> &'a [u8] {
        match self {
            SequenceColumnRef::String(arr) => arr.value(i).as_bytes(),
            SequenceColumnRef::LargeString(arr) => arr.value(i).as_bytes(),
        }
    }

    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            SequenceColumnRef::String(arr) => arr.is_null(i),
            SequenceColumnRef::LargeString(arr) => arr.is_null(i),
        }
    }
}

/// Extract a string column from a RecordBatch as SequenceColumnRef.
fn get_string_column<'a>(batch: &'a RecordBatch, col_name: &str) -> Result<SequenceColumnRef<'a>> {
    let col = batch
        .column_by_name(col_name)
        .ok_or_else(|| anyhow!("Missing {} column", col_name))?;

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(SequenceColumnRef::String(arr));
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(SequenceColumnRef::LargeString(arr));
    }

    Err(anyhow!(
        "{} column is not a string type: {:?}",
        col_name,
        col.data_type()
    ))
}

/// Convert a Parquet RecordBatch to QueryRecord references with zero-copy semantics.
///
/// This function creates batch-local indices as query_ids (0, 1, 2, ...) and
/// returns zero-copy references into the Arrow buffer memory for sequences.
///
/// # Arguments
/// * `batch` - RecordBatch from Parquet with 'sequence1' and optional 'sequence2' columns
///
/// # Returns
/// A vector of QueryRecord tuples with batch-local indices as IDs.
///
/// # Zero-Copy Guarantee
/// The returned sequence slices point directly into the Arrow buffers.
/// The batch must remain alive until classification is complete.
#[allow(dead_code)]
pub fn batch_to_records_parquet(batch: &RecordBatch) -> Result<Vec<QueryRecord<'_>>> {
    batch_to_records_parquet_with_offset(batch, 0)
}

/// Convert a Parquet RecordBatch to QueryRecord references with a starting index offset.
///
/// This is used when stacking multiple batches - each batch gets an offset so that
/// query_ids are globally unique across all stacked batches.
///
/// # Arguments
/// * `batch` - RecordBatch from Parquet with 'sequence1' and optional 'sequence2' columns
/// * `id_offset` - Starting index for query_ids in this batch
///
/// # Returns
/// A vector of QueryRecord tuples with offset indices as IDs.
pub fn batch_to_records_parquet_with_offset(
    batch: &RecordBatch,
    id_offset: usize,
) -> Result<Vec<QueryRecord<'_>>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Get sequence1 column
    let seq_col = get_string_column(batch, "sequence1")?;

    // Check if we have sequence2 column
    let pair_col = batch
        .column_by_name("sequence2")
        .map(|_| get_string_column(batch, "sequence2"))
        .transpose()?;

    // Build records with offset indices for stacking support
    let mut records = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        // Safe overflow checking: both usize addition and i64 conversion can fail
        let sum = id_offset
            .checked_add(i)
            .ok_or_else(|| anyhow!("Query ID overflow: offset {} + index {}", id_offset, i))?;
        let query_id =
            i64::try_from(sum).map_err(|_| anyhow!("Query ID {} exceeds i64::MAX", sum))?;
        let seq = seq_col.value(i);
        let pair = pair_col
            .as_ref()
            .and_then(|p| if p.is_null(i) { None } else { Some(p.value(i)) });
        records.push((query_id, seq, pair));
    }

    Ok(records)
}

/// Convert a RecordBatch to owned records with optional trimming.
///
/// This function copies sequences from the Arrow buffer into owned vectors,
/// optionally trimming them to a maximum length. This allows the RecordBatch
/// to be dropped immediately after conversion, freeing the Arrow buffer memory.
///
/// # Arguments
/// * `batch` - RecordBatch from Parquet with 'sequence1' and optional 'sequence2' columns
/// * `headers` - Pre-extracted headers for this batch
/// * `trim_to` - Optional maximum length for sequences. If provided, sequences longer
///   than this are truncated. Sequences with R1 shorter than trim_to are skipped.
/// * `id_offset` - Starting index for query_ids in this batch
///
/// # Returns
/// A tuple of:
/// - Owned records with copied (and optionally trimmed) sequences
/// - Headers for successfully converted records (skipped records are omitted)
///
/// # Memory Benefit
/// Unlike zero-copy conversion, this immediately copies only the needed data,
/// allowing the large Arrow buffer to be dropped. For long reads with small
/// `trim_to` values, this can reduce memory by 10-100x.
pub fn batch_to_owned_records_trimmed(
    batch: &RecordBatch,
    headers: &[String],
    trim_to: Option<usize>,
    minimum_length: Option<usize>,
    id_offset: usize,
) -> Result<(Vec<OwnedFastxRecord>, Vec<String>)> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Get sequence columns
    let seq_col = get_string_column(batch, "sequence1")?;

    let pair_col = batch
        .column_by_name("sequence2")
        .map(|_| get_string_column(batch, "sequence2"))
        .transpose()?;

    let mut records = Vec::with_capacity(num_rows);
    let mut out_headers = Vec::with_capacity(num_rows);

    // Using index loop because we need to access seq_col.value(i), pair_col.value(i),
    // and headers[i] - can't easily iterate all three together
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_rows {
        let seq1 = seq_col.value(i);

        // Skip if R1 is shorter than minimum_length (checked on original length)
        if let Some(min_len) = minimum_length {
            if seq1.len() < min_len {
                continue;
            }
        }

        // Skip if R1 is too short for trim_to requirement
        if let Some(trim_len) = trim_to {
            if seq1.len() < trim_len {
                continue; // Skip this record entirely
            }
        }

        // Calculate query_id using output record count (not input row index)
        let sum = id_offset.checked_add(records.len()).ok_or_else(|| {
            anyhow!(
                "Query ID overflow: offset {} + count {}",
                id_offset,
                records.len()
            )
        })?;
        let query_id =
            i64::try_from(sum).map_err(|_| anyhow!("Query ID {} exceeds i64::MAX", sum))?;

        // Copy and trim seq1
        let seq1_owned = match trim_to {
            Some(trim_len) => seq1[..trim_len.min(seq1.len())].to_vec(),
            None => seq1.to_vec(),
        };

        // Copy and trim seq2 if present
        let seq2_owned = pair_col.as_ref().and_then(|p| {
            if p.is_null(i) {
                None
            } else {
                let seq2 = p.value(i);
                match trim_to {
                    Some(trim_len) => Some(seq2[..trim_len.min(seq2.len())].to_vec()),
                    None => Some(seq2.to_vec()),
                }
            }
        });

        records.push(OwnedFastxRecord::new(
            query_id, seq1_owned, None, // qual1 - not available in Parquet input
            seq2_owned, None, // qual2 - not available in Parquet input
        ));
        out_headers.push(headers[i].clone());
    }

    debug_assert_eq!(
        records.len(),
        out_headers.len(),
        "Records and headers must stay synchronized"
    );

    Ok((records, out_headers))
}

/// Result of reading a batch of trimmed Parquet records.
pub struct TrimmedBatchResult {
    /// The accumulated owned records
    pub records: Vec<OwnedFastxRecord>,
    /// The corresponding headers
    pub headers: Vec<String>,
    /// Number of row groups processed
    pub rg_count: usize,
    /// Whether the end of input was reached
    pub reached_end: bool,
}

/// Accumulate pre-trimmed/filtered owned records from a Parquet reader.
///
/// This function reads `ParquetBatch::Owned` batches from the reader and
/// accumulates them until the target batch size is reached or end of input.
/// The reader thread has already applied trim_to and minimum_length filtering,
/// so this function only needs to remap query IDs for global uniqueness.
///
/// # Panics
/// Panics if the reader produces `ParquetBatch::Arrow` batches. This function
/// should only be called when trim/filter is active (which forces Owned output).
///
/// # Arguments
/// * `reader` - The Parquet reader (must have trim_to or minimum_length set)
/// * `target_batch_size` - Stop accumulating when this many records are collected
///
/// # Returns
/// A `TrimmedBatchResult` containing the accumulated records, headers, row group
/// count, and whether end of input was reached.
pub fn accumulate_owned_batches(
    reader: &mut PrefetchingParquetReader,
    target_batch_size: usize,
) -> Result<TrimmedBatchResult> {
    let mut records: Vec<OwnedFastxRecord> = Vec::new();
    let mut headers: Vec<String> = Vec::new();
    let mut reached_end = false;
    let mut rg_count = 0usize;

    while records.len() < target_batch_size {
        match reader.next_batch()? {
            Some(ParquetBatch::Owned(mut batch_records, batch_headers)) => {
                rg_count += 1;
                // Remap query_ids to be globally sequential across accumulated batches
                let offset = records.len() as i64;
                for rec in &mut batch_records {
                    rec.query_id += offset;
                }
                records.extend(batch_records);
                headers.extend(batch_headers);
            }
            Some(ParquetBatch::Arrow(..)) => {
                unreachable!("Expected Owned variant when trim/filter is active");
            }
            None => {
                reached_end = true;
                break;
            }
        }
    }

    Ok(TrimmedBatchResult {
        records,
        headers,
        rg_count,
        reached_end,
    })
}

/// Convert multiple stacked RecordBatches to a combined QueryRecord vector with zero-copy.
///
/// This function processes multiple batches together, assigning globally unique
/// query_ids across all batches. All batches must remain alive while the returned
/// records are in use.
///
/// # Arguments
/// * `batches` - Slice of (RecordBatch, headers) pairs to process together
///
/// # Returns
/// A tuple of:
/// - Combined QueryRecord vector with zero-copy references into all batches
/// - Combined headers vector
///
/// # Zero-Copy Guarantee
/// The returned sequence slices point directly into the Arrow buffers of the
/// respective batches. ALL batches in the input slice must remain alive until
/// classification is complete.
///
/// # Errors
/// - Returns an error if any batch is missing the required 'sequence1' column
/// - Returns an error if query ID calculation overflows (cumulative rows > i64::MAX)
/// - Returns an error if the 'sequence1' column is not a valid string array type
///
/// # Example
/// ```ignore
/// // Stack multiple batches for efficient parallel classification
/// let stacked: Vec<(RecordBatch, Vec<String>)> = collect_batches();
/// let (records, headers) = stacked_batches_to_records(&stacked)?;
///
/// // IMPORTANT: stacked must remain alive while records are in use
/// let results = index.classify_batch(&records, threshold, ...)?;
///
/// // Process results using headers for read IDs
/// for result in results {
///     let read_id = &headers[result.query_id as usize];
///     println!("{}\t{}", read_id, result.score);
/// }
/// // Now safe to drop stacked
/// drop(stacked);
/// ```
pub fn stacked_batches_to_records<'a>(
    batches: &'a [(RecordBatch, Vec<String>)],
) -> Result<(Vec<QueryRecord<'a>>, Vec<&'a str>)> {
    // Calculate total capacity
    let total_rows: usize = batches.iter().map(|(b, _)| b.num_rows()).sum();

    let mut all_records = Vec::with_capacity(total_rows);
    let mut all_headers = Vec::with_capacity(total_rows);
    let mut offset = 0usize;

    for (batch, headers) in batches {
        // Convert this batch with the current offset
        let records = batch_to_records_parquet_with_offset(batch, offset)?;
        all_records.extend(records);

        // Add headers as references (zero-copy for strings too)
        all_headers.extend(headers.iter().map(|s| s.as_str()));

        offset += batch.num_rows();
    }

    Ok((all_records, all_headers))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::LargeStringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::path::Path;
    use std::sync::Arc;

    // -------------------------------------------------------------------------
    // Tests for RowGroupWindow
    // -------------------------------------------------------------------------

    /// Spawns a thread blocked in `acquire()` and confirms it hasn't
    /// returned yet. Not asserting on exact timing -- the sleep is a
    /// generous margin to avoid a false pass by checking before the thread
    /// has had any chance to run at all.
    fn spawn_blocked_acquire(window: &Arc<RowGroupWindow>) -> std::thread::JoinHandle<bool> {
        let window = Arc::clone(window);
        let handle = std::thread::spawn(move || window.acquire());
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            !handle.is_finished(),
            "acquire() must block while the window is exhausted, not return immediately"
        );
        handle
    }

    #[test]
    fn test_row_group_window_blocks_when_exhausted_and_unblocks_on_release() {
        let window = Arc::new(RowGroupWindow::new(2));
        assert!(window.acquire());
        assert!(window.acquire());

        // Window now has 0 available permits; a third acquire on another
        // thread must block until a release happens.
        let handle = spawn_blocked_acquire(&window);

        window.release();
        assert!(
            handle.join().unwrap(),
            "acquire() must unblock and return true once a permit is released"
        );
    }

    #[test]
    fn test_row_group_window_abort_unblocks_waiters_without_a_permit() {
        let window = Arc::new(RowGroupWindow::new(1));
        assert!(window.acquire()); // exhaust the single permit

        let handle = spawn_blocked_acquire(&window);

        window.abort();
        assert!(
            !handle.join().unwrap(),
            "acquire() must return false (no permit) once aborted, not hang forever \
             waiting for a release that will never come"
        );
    }

    // -------------------------------------------------------------------------
    // Tests for is_parquet_input
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_parquet_input() {
        assert!(is_parquet_input(Path::new("input.parquet")));
        assert!(is_parquet_input(Path::new("input.PARQUET")));
        assert!(is_parquet_input(Path::new("/path/to/input.parquet")));
        assert!(!is_parquet_input(Path::new("input.fastq")));
        assert!(!is_parquet_input(Path::new("input.fasta")));
        assert!(!is_parquet_input(Path::new("input.parquet.gz")));
    }

    // -------------------------------------------------------------------------
    // Tests for batch_to_owned_records_trimmed
    // -------------------------------------------------------------------------

    /// Create a test RecordBatch with sequence data (uses LargeUtf8 like real Parquet files).
    fn make_test_batch(seqs: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sequence1",
            DataType::LargeUtf8,
            false,
        )]));

        let seq_array = LargeStringArray::from_iter_values(seqs.iter().copied());
        RecordBatch::try_new(schema, vec![Arc::new(seq_array)]).unwrap()
    }

    /// Create a test RecordBatch with paired sequences.
    fn make_test_batch_paired(seqs1: &[&str], seqs2: &[Option<&str>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence1", DataType::LargeUtf8, false),
            Field::new("sequence2", DataType::LargeUtf8, true),
        ]));

        let seq1_array = LargeStringArray::from_iter_values(seqs1.iter().copied());
        let seq2_array = LargeStringArray::from_iter(seqs2.iter().copied());
        RecordBatch::try_new(schema, vec![Arc::new(seq1_array), Arc::new(seq2_array)]).unwrap()
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_no_trim() {
        let seqs = vec!["ACGTACGTACGT", "GGGGCCCCAAAA"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["read1".to_string(), "read2".to_string()];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, None, None, 0).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(out_headers.len(), 2);
        assert_eq!(records[0].seq1, b"ACGTACGTACGT");
        assert_eq!(records[1].seq1, b"GGGGCCCCAAAA");
        assert_eq!(out_headers[0], "read1");
        assert_eq!(out_headers[1], "read2");
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_with_trim() {
        let seqs = vec!["ACGTACGTACGT", "GGGGCCCCAAAA"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["read1".to_string(), "read2".to_string()];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(4), None, 0).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(out_headers.len(), 2);
        // Should be trimmed to first 4 bases
        assert_eq!(records[0].seq1, b"ACGT");
        assert_eq!(records[1].seq1, b"GGGG");
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_skip_short_reads() {
        // One long read (12bp) and one short read (4bp)
        let seqs = vec!["ACGTACGTACGT", "GGGG"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["long_read".to_string(), "short_read".to_string()];

        // Trim to 8bp - should skip the 4bp read
        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(8), None, 0).unwrap();

        assert_eq!(records.len(), 1, "Short read should be skipped");
        assert_eq!(out_headers.len(), 1);
        assert_eq!(records[0].seq1, b"ACGTACGT");
        assert_eq!(out_headers[0], "long_read");
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_query_id_with_offset() {
        let seqs = vec!["ACGTACGTACGT", "GGGGCCCCAAAA"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["read1".to_string(), "read2".to_string()];

        // Start with offset 100
        let (records, _) =
            batch_to_owned_records_trimmed(&batch, &headers, None, None, 100).unwrap();

        assert_eq!(records[0].query_id, 100, "First query_id should be offset");
        assert_eq!(
            records[1].query_id, 101,
            "Second query_id should be offset+1"
        );
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_query_id_with_skipped_reads() {
        // Mix of long and short reads
        let seqs = vec!["ACGTACGTACGT", "GG", "TTTTTTTTTTTT"];
        let batch = make_test_batch(&seqs);
        let headers = vec![
            "long1".to_string(),
            "short".to_string(),
            "long2".to_string(),
        ];

        // Trim to 8bp - middle read should be skipped
        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(8), None, 0).unwrap();

        assert_eq!(records.len(), 2);
        // Query IDs should be sequential based on OUTPUT count, not input row
        assert_eq!(records[0].query_id, 0);
        assert_eq!(records[1].query_id, 1);
        assert_eq!(out_headers[0], "long1");
        assert_eq!(out_headers[1], "long2");
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_paired_sequences() {
        let seqs1 = vec!["ACGTACGTACGT", "GGGGCCCCAAAA"];
        let seqs2: Vec<Option<&str>> = vec![Some("TTTTTTTTTTTT"), Some("CCCCCCCCCCCC")];
        let batch = make_test_batch_paired(&seqs1, &seqs2);
        let headers = vec!["read1".to_string(), "read2".to_string()];

        let (records, _) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(4), None, 0).unwrap();

        assert_eq!(records.len(), 2);
        // Both R1 and R2 should be trimmed
        assert_eq!(records[0].seq1, b"ACGT");
        assert_eq!(records[0].seq2.as_ref().unwrap(), b"TTTT");
        assert_eq!(records[1].seq1, b"GGGG");
        assert_eq!(records[1].seq2.as_ref().unwrap(), b"CCCC");
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_empty_batch() {
        let seqs: Vec<&str> = vec![];
        let batch = make_test_batch(&seqs);
        let headers: Vec<String> = vec![];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(100), None, 0).unwrap();

        assert!(records.is_empty());
        assert!(out_headers.is_empty());
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_all_reads_too_short() {
        let seqs = vec!["ACGT", "GGGG", "TTTT"];
        let batch = make_test_batch(&seqs);
        let headers = vec![
            "read1".to_string(),
            "read2".to_string(),
            "read3".to_string(),
        ];

        // Trim to 100bp - all reads are shorter
        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(100), None, 0).unwrap();

        assert!(records.is_empty(), "All reads should be skipped");
        assert!(out_headers.is_empty());
    }

    #[test]
    fn test_batch_to_owned_records_trimmed_records_headers_synchronized() {
        // This test verifies the debug_assert is correct
        let seqs = vec!["ACGTACGTACGT", "GG", "TTTTTTTTTTTT", "AA"];
        let batch = make_test_batch(&seqs);
        let headers = vec![
            "keep1".to_string(),
            "skip1".to_string(),
            "keep2".to_string(),
            "skip2".to_string(),
        ];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(8), None, 0).unwrap();

        // Records and headers must have same length
        assert_eq!(
            records.len(),
            out_headers.len(),
            "Records and headers must be synchronized"
        );

        // Verify the kept reads match the kept headers
        assert_eq!(out_headers[0], "keep1");
        assert_eq!(out_headers[1], "keep2");
    }

    // -------------------------------------------------------------------------
    // Tests for batch_to_owned_records_trimmed with minimum_length
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_to_owned_records_with_minimum_length() {
        // 3 reads: 30bp, 80bp, 50bp; min_length=50 → 2 records (30bp skipped)
        let s0 = "A".repeat(30);
        let s1 = "G".repeat(80);
        let s2 = "T".repeat(50);
        let seqs_ref = vec![s0.as_str(), s1.as_str(), s2.as_str()];
        let batch = make_test_batch(&seqs_ref);
        let headers = vec![
            "short30".to_string(),
            "long80".to_string(),
            "exact50".to_string(),
        ];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, None, Some(50), 0).unwrap();

        assert_eq!(records.len(), 2, "30bp read should be skipped");
        assert_eq!(out_headers.len(), 2);
        assert_eq!(records[0].seq1.len(), 80);
        assert_eq!(records[1].seq1.len(), 50);
        assert_eq!(out_headers[0], "long80");
        assert_eq!(out_headers[1], "exact50");
    }

    #[test]
    fn test_batch_to_owned_records_minimum_length_before_trim() {
        // min_length=50, trim_to=70:
        //   40bp → skipped by min_length (40 < 50)
        //   100bp → passes min_length, passes trim_to, trimmed to 70
        //   60bp → passes min_length (60 >= 50), but skipped by trim_to (60 < 70)
        //   80bp → passes both, trimmed to 70
        let s0 = "A".repeat(40);
        let s1 = "G".repeat(100);
        let s2 = "T".repeat(60);
        let s3 = "C".repeat(80);
        let seqs_ref = vec![s0.as_str(), s1.as_str(), s2.as_str(), s3.as_str()];
        let batch = make_test_batch(&seqs_ref);
        let headers = vec![
            "short40".to_string(),
            "long100".to_string(),
            "mid60".to_string(),
            "mid80".to_string(),
        ];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(70), Some(50), 0).unwrap();

        // 40bp skipped by min_length, 60bp skipped by trim_to, 100bp and 80bp kept
        assert_eq!(records.len(), 2);
        assert_eq!(out_headers[0], "long100");
        assert_eq!(out_headers[1], "mid80");
        assert_eq!(records[0].seq1.len(), 70, "100bp trimmed to 70");
        assert_eq!(records[1].seq1.len(), 70, "80bp trimmed to 70");
    }

    #[test]
    fn test_batch_to_owned_records_minimum_length_with_paired() {
        // Pair skipped when R1 < min_length, even if R2 is long
        let s1_short = "A".repeat(30);
        let s1_long = "G".repeat(80);
        let s2_long = "T".repeat(100);
        let s2_short = "C".repeat(20);
        let seqs1 = vec![s1_short.as_str(), s1_long.as_str()];
        let seqs2: Vec<Option<&str>> = vec![Some(s2_long.as_str()), Some(s2_short.as_str())];
        let batch = make_test_batch_paired(&seqs1, &seqs2);
        let headers = vec!["pair_short_r1".to_string(), "pair_long_r1".to_string()];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, None, Some(50), 0).unwrap();

        assert_eq!(records.len(), 1, "Pair with R1=30bp should be skipped");
        assert_eq!(out_headers[0], "pair_long_r1");
        assert_eq!(records[0].seq1.len(), 80);
        assert_eq!(records[0].seq2.as_ref().unwrap().len(), 20);
    }

    #[test]
    fn test_batch_to_owned_records_minimum_length_gt_trim_to() {
        // min_length=100, trim_to=50: only reads >= 100bp kept, trimmed to 50
        let s0 = "A".repeat(80); // < 100, skipped
        let s1 = "G".repeat(120); // >= 100, kept and trimmed to 50
        let s2 = "T".repeat(100); // >= 100, kept and trimmed to 50
        let s3 = "C".repeat(40); // < 100, skipped
        let seqs_ref = vec![s0.as_str(), s1.as_str(), s2.as_str(), s3.as_str()];
        let batch = make_test_batch(&seqs_ref);
        let headers = vec![
            "r80".to_string(),
            "r120".to_string(),
            "r100".to_string(),
            "r40".to_string(),
        ];

        let (records, out_headers) =
            batch_to_owned_records_trimmed(&batch, &headers, Some(50), Some(100), 0).unwrap();

        assert_eq!(records.len(), 2, "Only reads >= 100bp should survive");
        assert_eq!(out_headers[0], "r120");
        assert_eq!(out_headers[1], "r100");
        assert_eq!(records[0].seq1.len(), 50, "120bp trimmed to 50");
        assert_eq!(records[1].seq1.len(), 50, "100bp trimmed to 50");
        // Query IDs should be sequential
        assert_eq!(records[0].query_id, 0);
        assert_eq!(records[1].query_id, 1);
    }

    // -------------------------------------------------------------------------
    // Tests for ParquetBatch enum
    // -------------------------------------------------------------------------

    #[test]
    fn test_parquet_batch_enum_arrow_into_arrow() {
        let seqs = vec!["ACGTACGT"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["read1".to_string()];

        let pb = ParquetBatch::Arrow(batch, headers);
        let (record_batch, hdrs) = pb.into_arrow();

        assert_eq!(record_batch.num_rows(), 1);
        assert_eq!(hdrs.len(), 1);
        assert_eq!(hdrs[0], "read1");
    }

    #[test]
    fn test_parquet_batch_enum_owned_into_owned() {
        let records = vec![OwnedFastxRecord::new(
            0,
            b"ACGTACGT".to_vec(),
            None,
            None,
            None,
        )];
        let headers = vec!["read1".to_string()];

        let pb = ParquetBatch::Owned(records, headers);
        let (owned_records, hdrs) = pb.into_owned();

        assert_eq!(owned_records.len(), 1);
        assert_eq!(owned_records[0].seq1, b"ACGTACGT");
        assert_eq!(hdrs.len(), 1);
        assert_eq!(hdrs[0], "read1");
    }

    #[test]
    #[should_panic(expected = "Expected ParquetBatch::Arrow but got Owned")]
    fn test_parquet_batch_enum_owned_into_arrow_panics() {
        let records = vec![OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None)];
        let headers = vec!["read1".to_string()];

        let pb = ParquetBatch::Owned(records, headers);
        let _ = pb.into_arrow(); // should panic
    }

    #[test]
    #[should_panic(expected = "Expected ParquetBatch::Owned but got Arrow")]
    fn test_parquet_batch_enum_arrow_into_owned_panics() {
        let seqs = vec!["ACGT"];
        let batch = make_test_batch(&seqs);
        let headers = vec!["read1".to_string()];

        let pb = ParquetBatch::Arrow(batch, headers);
        let _ = pb.into_owned(); // should panic
    }

    // -------------------------------------------------------------------------
    // Tests for PrefetchingParquetReader trim/filter in reader thread
    // -------------------------------------------------------------------------

    use parquet::arrow::ArrowWriter;
    use tempfile::tempdir;

    /// Write a Parquet file with read_id + sequence1 columns (the schema
    /// PrefetchingParquetReader expects).
    fn write_test_parquet(dir: &std::path::Path, ids: &[&str], seqs: &[&str]) -> PathBuf {
        let path = dir.join("test.parquet");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("read_id", DataType::LargeUtf8, false),
            Field::new("sequence1", DataType::LargeUtf8, false),
        ]));

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        let id_array = LargeStringArray::from_iter_values(ids.iter().copied());
        let seq_array = LargeStringArray::from_iter_values(seqs.iter().copied());
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    #[test]
    fn test_prefetching_parquet_reader_trims_in_reader_thread() {
        let dir = tempdir().unwrap();
        let s0 = "A".repeat(30);
        let s1 = "G".repeat(100);
        let s2 = "T".repeat(60);
        let path = write_test_parquet(dir.path(), &["r0", "r1", "r2"], &[&s0, &s1, &s2]);

        // trim_to=50, minimum_length=40 → r0 (30bp) skipped by min_length,
        // r1 (100bp) trimmed to 50, r2 (60bp) trimmed to 50
        let mut reader = PrefetchingParquetReader::with_parallel_row_groups(
            &path,
            1000,
            None, // sequential
            Some(50),
            Some(40),
        )
        .unwrap();

        let mut all_records = Vec::new();
        let mut all_headers = Vec::new();
        while let Some(batch) = reader.next_batch().unwrap() {
            // Must be Owned variant when trim/filter active
            let (records, headers) = batch.into_owned();
            all_records.extend(records);
            all_headers.extend(headers);
        }

        assert_eq!(all_records.len(), 2, "30bp read should be filtered out");
        assert_eq!(all_headers.len(), 2);

        assert_eq!(all_headers[0], "r1");
        assert_eq!(all_headers[1], "r2");
        assert_eq!(
            all_records[0].seq1.len(),
            50,
            "100bp should be trimmed to 50"
        );
        assert_eq!(
            all_records[1].seq1.len(),
            50,
            "60bp should be trimmed to 50"
        );

        reader.finish().unwrap();
    }

    #[test]
    fn test_prefetching_parquet_reader_parallel_trims_in_reader_thread() {
        let dir = tempdir().unwrap();
        let s0 = "A".repeat(30);
        let s1 = "G".repeat(100);
        let s2 = "T".repeat(60);
        let path = write_test_parquet(dir.path(), &["r0", "r1", "r2"], &[&s0, &s1, &s2]);

        // Same trim/filter as sequential test but with parallel_row_groups=Some(2)
        let mut reader = PrefetchingParquetReader::with_parallel_row_groups(
            &path,
            1000,
            Some(2), // parallel
            Some(50),
            Some(40),
        )
        .unwrap();

        let mut all_records = Vec::new();
        let mut all_headers = Vec::new();
        while let Some(batch) = reader.next_batch().unwrap() {
            let (records, headers) = batch.into_owned();
            all_records.extend(records);
            all_headers.extend(headers);
        }

        assert_eq!(all_records.len(), 2, "30bp read should be filtered out");
        assert_eq!(all_headers.len(), 2);

        assert_eq!(all_headers[0], "r1");
        assert_eq!(all_headers[1], "r2");
        assert_eq!(all_records[0].seq1.len(), 50);
        assert_eq!(all_records[1].seq1.len(), 50);

        reader.finish().unwrap();
    }

    /// Write a Parquet file with one row group per record (forcing a
    /// boundary via `flush()` after each write), so tests can exercise
    /// cross-row-group ordering with `parallel_row_groups` set below the
    /// row group count.
    fn write_test_parquet_multi_rg(dir: &std::path::Path, ids: &[&str], seqs: &[&str]) -> PathBuf {
        let path = dir.join("test_multi_rg.parquet");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("read_id", DataType::LargeUtf8, false),
            Field::new("sequence1", DataType::LargeUtf8, false),
        ]));

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        for (id, seq) in ids.iter().zip(seqs.iter()) {
            let id_array = LargeStringArray::from_iter_values([*id]);
            let seq_array = LargeStringArray::from_iter_values([*seq]);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(id_array), Arc::new(seq_array)],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap(); // force a row-group boundary per record
        }

        writer.close().unwrap();
        path
    }

    #[test]
    fn test_prefetching_parquet_reader_parallel_preserves_order_across_many_row_groups() {
        let dir = tempdir().unwrap();
        let ids: Vec<String> = (0..10).map(|i| format!("r{i}")).collect();
        let seqs: Vec<String> = (0..10).map(|i| "ACGT".repeat(i + 1)).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
        let path = write_test_parquet_multi_rg(dir.path(), &id_refs, &seq_refs);

        // 10 row groups, parallelism of 3 -- under the old step_by(parallel_rg)
        // chunking this forces uneven waves (3+3+3+1); it exercises the
        // bounded-queue reorder buffer's uneven-leftover path the same way.
        let mut reader =
            PrefetchingParquetReader::with_parallel_row_groups(&path, 1000, Some(3), None, None)
                .unwrap();

        let mut all_headers = Vec::new();
        while let Some(batch) = reader.next_batch().unwrap() {
            let (record_batch, headers) = batch.into_arrow();
            assert_eq!(record_batch.num_rows(), headers.len());
            all_headers.extend(headers);
        }

        assert_eq!(
            all_headers, ids,
            "row groups must be emitted in original file order regardless of read concurrency"
        );
        reader.finish().unwrap();
    }

    #[test]
    fn test_prefetching_parquet_reader_no_filter_returns_arrow() {
        let dir = tempdir().unwrap();
        let path = write_test_parquet(
            dir.path(),
            &["r0", "r1", "r2"],
            &["ACGTACGT", "GGGGCCCC", "TTTTAAAA"],
        );

        // No trim/filter → Arrow variant (zero-copy preserved)
        let mut reader =
            PrefetchingParquetReader::with_parallel_row_groups(&path, 1000, None, None, None)
                .unwrap();

        let mut total_rows = 0;
        while let Some(batch) = reader.next_batch().unwrap() {
            // Must be Arrow variant when no trim/filter
            let (record_batch, headers) = batch.into_arrow();
            total_rows += record_batch.num_rows();
            assert_eq!(record_batch.num_rows(), headers.len());
        }

        assert_eq!(total_rows, 3);
        reader.finish().unwrap();
    }

    /// Regression test for a deadlock: `reader_thread_parallel`'s row-group
    /// workers must not be scheduled as rayon tasks on the shared global
    /// pool. Each worker runs a `loop` that never returns control until all
    /// row groups are claimed, so when they *were* `rayon::scope` tasks,
    /// setting `parallel_rg >= rayon::current_num_threads()` let them
    /// permanently occupy every pool worker for the whole read -- starving
    /// any *other* rayon caller sharing the pool. In production that other
    /// caller is `process_batch()`'s own `par_iter()` extraction on the
    /// consumer thread, interleaved between `next_batch()` calls exactly as
    /// this test does; it would then block forever waiting for a worker
    /// rayon can never free up, since the reader's own workers never yield.
    /// Reproduced as a deterministic hang via the real CLI before the fix
    /// (switching the workers to plain `std::thread::spawn` OS threads,
    /// which don't compete with rayon's pool at all).
    #[test]
    fn test_parallel_reader_does_not_starve_concurrent_rayon_work() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("oversubscribed.parquet");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("read_id", DataType::LargeUtf8, false),
            Field::new("sequence1", DataType::LargeUtf8, false),
        ]));
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        let num_threads = rayon::current_num_threads();
        // Enough row groups to keep every worker busy for a few rounds, with
        // real (non-instant) per-row-group work, so the concurrent rayon
        // call below has a real chance to be attempted while the read is
        // still in flight rather than after it has already finished.
        let num_row_groups = num_threads * 4;
        for rg in 0..num_row_groups {
            let ids: Vec<String> = (0..500).map(|i| format!("rg{}_r{}", rg, i)).collect();
            let seqs: Vec<String> = (0..500).map(|_| "A".repeat(300)).collect();
            let id_array = LargeStringArray::from_iter_values(ids.iter().map(|s| s.as_str()));
            let seq_array = LargeStringArray::from_iter_values(seqs.iter().map(|s| s.as_str()));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(id_array), Arc::new(seq_array)],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
        let total = num_row_groups * 500;

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            // Deliberately oversubscribed relative to the pool, matching
            // the shape that deadlocked in production.
            let mut reader = PrefetchingParquetReader::with_parallel_row_groups(
                &path,
                1000,
                Some(num_threads + 4),
                None,
                None,
            )
            .unwrap();
            let mut count = 0;
            while let Some(batch) = reader.next_batch().unwrap() {
                let (rb, _) = batch.into_arrow();
                count += rb.num_rows();
                // Mirrors classify.rs's process_batch(): real rayon work on
                // the consumer thread, interleaved with reads.
                use rayon::prelude::*;
                let _: u64 = (0..10_000u64).into_par_iter().map(|x| x * x).sum();
            }
            reader.finish().unwrap();
            let _ = done_tx.send(count);
        });

        match done_rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(count) => assert_eq!(count, total),
            Err(_) => panic!(
                "HUNG: parallel reader starved concurrent rayon work \
                 (did not finish within 30s)"
            ),
        }
    }

    /// Generates `scratch/bench-uneven-rg-large.parquet`: 104 row groups,
    /// mostly small (2,000 short reads each) but with a much larger
    /// "straggler" row group (20,000 longer reads) at every 8th position --
    /// matching the old `step_by(parallel_rg=8)` chunking, so every chunk
    /// pays a "wait for the slow one" barrier under the old lockstep design.
    /// This is the scenario Phase 4's bounded-work-queue rewrite targets:
    /// letting workers race ahead into later chunks' fast row groups instead
    /// of idling at each synthetic chunk boundary. Not run in CI; regenerate
    /// manually before benchmarking `--parallel-input-rg` with
    /// `cargo test --release -- --ignored gen_bench_fixture`.
    #[test]
    #[ignore]
    fn gen_bench_fixture_uneven_row_groups() {
        let dir = Path::new("scratch");
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("bench-uneven-rg-large.parquet");

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("read_id", DataType::LargeUtf8, false),
            Field::new("sequence1", DataType::LargeUtf8, false),
        ]));
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        // Deterministic pseudo-random ACGT sequence, no `rand` dependency needed.
        fn make_seq(seed: usize, len: usize) -> String {
            const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
            let mut state = seed as u64 ^ 0x9E3779B97F4A7C15;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    BASES[(state % 4) as usize] as char
                })
                .collect()
        }

        let mut write_row_group = |start_id: usize, count: usize, seq_len: usize| {
            let ids: Vec<String> = (0..count).map(|i| format!("r{}", start_id + i)).collect();
            let seqs: Vec<String> = (0..count)
                .map(|i| make_seq(start_id + i, seq_len))
                .collect();
            let id_array = LargeStringArray::from_iter_values(ids.iter().map(|s| s.as_str()));
            let seq_array = LargeStringArray::from_iter_values(seqs.iter().map(|s| s.as_str()));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(id_array), Arc::new(seq_array)],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap();
        };

        // A straggler in *every* chunk-sized window (matching the old
        // step_by(parallel_rg=8) chunking): this is the pattern where the
        // old lockstep design pays a "wait for the slow one" barrier on
        // every single chunk, while the bounded-window design lets workers
        // race ahead into later chunks' fast row groups instead of idling
        // at each synthetic chunk boundary.
        let mut next_id = 0usize;
        for rg in 0..104 {
            if rg % 8 == 0 {
                write_row_group(next_id, 20_000, 500); // straggler
                next_id += 20_000;
            } else {
                write_row_group(next_id, 2_000, 150);
                next_id += 2_000;
            }
        }
        writer.close().unwrap();

        eprintln!("wrote {} ({} total reads)", path.display(), next_id);
    }
}
