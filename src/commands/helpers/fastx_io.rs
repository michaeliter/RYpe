//! FASTX (FASTA/FASTQ) I/O with background prefetching.

use anyhow::{anyhow, Result};
use flate2::read::MultiGzDecoder;
use needletail::errors::ParseError;
use needletail::parser::{FastaReader, FastqReader};
use needletail::{parse_fastx_reader, FastxReader};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::FirstErrorCapture;

use super::output::{OutputFormat, OutputWriter};

/// Owned FASTX record with sequence and quality data.
///
/// Headers are stored separately in the `Vec<String>` returned alongside records
/// from batch readers, avoiding duplicate storage. Use the batch-local `query_id`
/// to index into the headers vector when writing output.
///
/// Quality scores are only captured when `preserve_for_output` is enabled in the reader.
#[derive(Debug, Clone)]
pub struct OwnedFastxRecord {
    pub query_id: i64,
    pub seq1: Vec<u8>,
    pub qual1: Option<Vec<u8>>,
    pub seq2: Option<Vec<u8>>,
    pub qual2: Option<Vec<u8>>,
}

impl OwnedFastxRecord {
    /// Create a new owned FASTX record.
    pub fn new(
        query_id: i64,
        seq1: Vec<u8>,
        qual1: Option<Vec<u8>>,
        seq2: Option<Vec<u8>>,
        qual2: Option<Vec<u8>>,
    ) -> Self {
        Self {
            query_id,
            seq1,
            qual1,
            seq2,
            qual2,
        }
    }

    /// Returns true if this is a FASTQ record (has quality scores).
    #[allow(dead_code)]
    pub fn is_fastq(&self) -> bool {
        self.qual1.is_some()
    }

    /// Returns true if this is a paired-end record.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.seq2.is_some()
    }

    /// Get references to the sequences.
    #[allow(dead_code)]
    pub fn sequences(&self) -> (&[u8], Option<&[u8]>) {
        (&self.seq1, self.seq2.as_deref())
    }
}

/// Type alias for batch data sent through the prefetch channel.
type BatchResult = Result<Option<(Vec<OwnedFastxRecord>, Vec<String>)>>;

/// Default timeout for waiting on prefetch batches (5 minutes).
const DEFAULT_PREFETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// I/O handler with background prefetching for overlapping I/O and computation.
///
/// This handler spawns a background thread that reads and decompresses the next
/// batch while the main thread processes the current batch. This is especially
/// effective for gzipped input where decompression can be overlapped with
/// minimizer extraction and classification. Each gzipped input file is
/// additionally inflated on its own thread (see [`open_fastx`]).
///
/// # Usage
/// ```ignore
/// let mut io = PrefetchingIoHandler::new(&r1, r2.as_ref(), output, batch_size)?;
/// while let Some((records, headers)) = io.next_batch()? {
///     // Process batch...
/// }
/// io.finish()?;
/// ```
pub struct PrefetchingIoHandler {
    receiver: Receiver<BatchResult>,
    prefetch_thread: Option<JoinHandle<()>>,
    writer: OutputWriter,
    /// Thread-safe error capture for errors that occur when sender.send() fails.
    /// Only captures the first error to avoid race conditions.
    error_capture: Arc<FirstErrorCapture>,
    /// Timeout for waiting on batches
    timeout: Duration,
}

impl PrefetchingIoHandler {
    /// Create a new prefetching I/O handler.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None). Format auto-detected from extension:
    ///   - `.tsv` or no extension: Plain TSV
    ///   - `.tsv.gz`: Gzip-compressed TSV
    ///   - `.parquet`: Apache Parquet with zstd compression
    ///   - `-`: stdout (TSV)
    /// * `batch_size` - Number of records per batch
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    #[allow(dead_code)]
    pub fn new(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
    ) -> Result<Self> {
        Self::with_trim(r1_path, r2_path, out_path, batch_size, None)
    }

    /// Create a new prefetching I/O handler with optional sequence trimming.
    ///
    /// When `trim_to` is specified, sequences are trimmed at read time to reduce
    /// memory usage for long reads. Records with R1 shorter than `trim_to` are skipped.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None)
    /// * `batch_size` - Number of records per batch
    /// * `trim_to` - Optional maximum sequence length. Sequences longer than this are
    ///   truncated at read time. Records with R1 shorter than this are skipped.
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    pub fn with_trim(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
    ) -> Result<Self> {
        Self::with_options(r1_path, r2_path, out_path, batch_size, trim_to, None, false)
    }

    /// Create a new prefetching I/O handler with full options.
    ///
    /// This constructor provides control over all reader options including
    /// quality score preservation for sequence output.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None)
    /// * `batch_size` - Number of records per batch
    /// * `trim_to` - Optional maximum sequence length
    /// * `minimum_length` - Optional minimum R1 length filter. Applied before `trim_to`:
    ///   reads with R1 shorter than this are skipped entirely (not modified).
    /// * `preserve_for_output` - When true, capture quality scores for FASTQ output.
    ///   Only enable when using `--output-sequences` to avoid wasting memory.
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    pub fn with_options(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
        preserve_for_output: bool,
    ) -> Result<Self> {
        // Clone paths for the background thread
        let r1_path = r1_path.to_path_buf();
        let r2_path = r2_path.cloned();

        // Thread-safe error capture (only stores first error)
        let error_capture = Arc::new(FirstErrorCapture::new());
        let thread_error = Arc::clone(&error_capture);

        // Use sync_channel with buffer of 2 to allow actual prefetching:
        // - Slot 1: batch currently being processed by main thread
        // - Slot 2: next batch being prefetched by reader thread
        let (sender, receiver): (SyncSender<BatchResult>, Receiver<BatchResult>) =
            mpsc::sync_channel(2);

        // Spawn background thread for reading
        let prefetch_thread = thread::spawn(move || {
            Self::reader_thread(
                r1_path,
                r2_path,
                batch_size,
                trim_to,
                minimum_length,
                preserve_for_output,
                sender,
                thread_error,
            );
        });

        // Set up output writer with auto-detected format
        let output_format = OutputFormat::detect(out_path.as_ref());
        let writer = OutputWriter::new(output_format, out_path.as_ref(), None)?;

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            writer,
            error_capture,
            timeout: DEFAULT_PREFETCH_TIMEOUT,
        })
    }

    /// Extract the base read ID (before any space, tab, or /1 /2 suffix).
    ///
    /// This handles common FASTQ header formats:
    /// - `@READ1 comment` → `READ1`
    /// - `@READ1/1` → `READ1`
    /// - `@READ1/1 comment` → `READ1`
    /// - `@READ1\tcomment` → `READ1`
    ///
    /// Note: Only `/1` and `/2` suffixes at the end of the ID portion (before space/tab)
    /// are stripped. The function assumes ASCII input (standard for FASTQ).
    pub fn base_read_id(id: &[u8]) -> &[u8] {
        // Find first space or tab - everything after is a comment
        let id_end = id
            .iter()
            .position(|&b| b == b' ' || b == b'\t')
            .unwrap_or(id.len());

        let id_portion = &id[..id_end];

        // Check for /1 or /2 suffix at the end of the ID portion
        if id_portion.len() >= 2 {
            let len = id_portion.len();
            if id_portion[len - 2] == b'/'
                && (id_portion[len - 1] == b'1' || id_portion[len - 1] == b'2')
            {
                return &id_portion[..len - 2];
            }
        }

        id_portion
    }

    /// Background thread function that reads batches and sends them through the channel.
    ///
    /// When `trim_to` is specified:
    /// - Records with R1 shorter than `trim_to` are skipped entirely
    /// - R1 sequences are truncated to `trim_to` length
    /// - R2 sequences are truncated to `min(len, trim_to)` (never skip based on R2)
    ///
    /// When `preserve_for_output` is true:
    /// - Quality scores are captured for FASTQ files (needed for `--output-sequences`)
    /// - This uses more memory, so only enable when writing sequences out
    #[allow(clippy::too_many_arguments)]
    fn reader_thread(
        r1_path: PathBuf,
        r2_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
        minimum_length: Option<usize>,
        preserve_for_output: bool,
        sender: SyncSender<BatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
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

        // Open readers in the background thread
        let mut r1 = match open_fastx(&r1_path) {
            Ok(r) => r,
            Err(e) => {
                send_error!(format!("Failed to open R1 at {}: {}", r1_path.display(), e));
            }
        };

        let mut r2: Option<Box<dyn FastxReader>> = match r2_path {
            Some(ref p) => match open_fastx(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    send_error!(format!("Failed to open R2 at {}: {}", p.display(), e));
                }
            },
            None => None,
        };

        let mut global_record_id: i64 = 0; // For error messages only

        // Macro for skipping a record: consume R2 if paired to keep files in sync,
        // increment global_record_id, and continue to the next record.
        macro_rules! skip_record {
            () => {{
                if let Some(ref mut r2_reader) = r2 {
                    match r2_reader.next() {
                        Some(Ok(_)) => {}
                        Some(Err(e)) => {
                            send_error!(format!(
                                "Error reading R2 at record {}: {}",
                                global_record_id, e
                            ));
                        }
                        None => {
                            send_error!(format!(
                                "R1/R2 mismatch: R2 ended early at record {}",
                                global_record_id
                            ));
                        }
                    }
                }
                global_record_id += 1;
                continue;
            }};
        }

        loop {
            let mut records = Vec::with_capacity(batch_size);
            let mut headers = Vec::with_capacity(batch_size);

            while records.len() < batch_size {
                let s1_rec = match r1.next() {
                    Some(Ok(rec)) => rec,
                    Some(Err(e)) => {
                        send_error!(format!(
                            "Error reading R1 at record {}: {}",
                            global_record_id, e
                        ));
                    }
                    None => break, // End of file
                };

                // Get R1 sequence
                let s1_seq = s1_rec.seq();

                // Check minimum_length on original R1 length (before trim_to)
                if let Some(min_len) = minimum_length {
                    if s1_seq.len() < min_len {
                        skip_record!();
                    }
                }

                // Check trim_to requirement on original R1 length
                if let Some(trim_len) = trim_to {
                    if s1_seq.len() < trim_len {
                        skip_record!();
                    }
                }

                // Helper to trim and copy a byte slice
                let trim_copy = |data: &[u8], trim_to: Option<usize>| -> Vec<u8> {
                    match trim_to {
                        Some(trim_len) => data[..trim_len.min(data.len())].to_vec(),
                        None => data.to_vec(),
                    }
                };

                // Copy R1 sequence with optional trimming
                let s1_vec = trim_copy(&s1_seq, trim_to);

                // Capture R1 quality if preserving for output and this is FASTQ
                let q1_vec = if preserve_for_output {
                    s1_rec.qual().map(|q| trim_copy(q, trim_to))
                } else {
                    None
                };

                // Handle R2 if present
                let (s2_vec, q2_vec) = if let Some(ref mut r2_reader) = r2 {
                    match r2_reader.next() {
                        Some(Ok(rec)) => {
                            // Validate that read IDs match
                            let r1_base = Self::base_read_id(s1_rec.id());
                            let r2_base = Self::base_read_id(rec.id());
                            if r1_base != r2_base {
                                send_error!(format!(
                                    "R1/R2 read ID mismatch at record {}: R1='{}' R2='{}'",
                                    global_record_id,
                                    String::from_utf8_lossy(r1_base),
                                    String::from_utf8_lossy(r2_base)
                                ));
                            }
                            // Copy R2 sequence with optional trimming
                            let s2_seq = rec.seq();
                            let s2 = Some(trim_copy(&s2_seq, trim_to));

                            // Capture R2 quality if preserving for output
                            let q2 = if preserve_for_output {
                                rec.qual().map(|q| trim_copy(q, trim_to))
                            } else {
                                None
                            };

                            (s2, q2)
                        }
                        Some(Err(e)) => {
                            send_error!(format!(
                                "Error reading R2 at record {}: {}",
                                global_record_id, e
                            ));
                        }
                        None => {
                            send_error!(format!(
                                "R1/R2 mismatch: R2 ended early at record {}",
                                global_record_id
                            ));
                        }
                    }
                } else {
                    (None, None)
                };

                let base_id = Self::base_read_id(s1_rec.id());
                let header = String::from_utf8_lossy(base_id).to_string();
                // Use batch-local index (0-based within each batch) for query_id.
                // This is intentional - the classification code maps results back
                // to headers using this batch-local index.
                records.push(OwnedFastxRecord::new(
                    records.len() as i64,
                    s1_vec,
                    q1_vec,
                    s2_vec,
                    q2_vec,
                ));
                headers.push(header);
                global_record_id += 1;
            }

            if records.is_empty() {
                // End of input - send None to signal completion
                let _ = sender.send(Ok(None));
                return;
            }

            // Send the batch - this will block if the channel is full (backpressure)
            if sender.send(Ok(Some((records, headers)))).is_err() {
                // Receiver was dropped - not an error, just exit cleanly
                return;
            }
        }
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some((records, headers)))` for each batch,
    /// `Ok(None)` when all records have been read,
    /// or `Err` if an error occurred during reading.
    ///
    /// Uses a timeout (default 5 minutes) to avoid hanging indefinitely
    /// if the reader thread stalls (e.g., NFS mount issues, disk errors).
    pub fn next_batch(&mut self) -> Result<Option<(Vec<OwnedFastxRecord>, Vec<String>)>> {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                // Check if there's a stored error from the reader thread
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                Err(anyhow!(
                    "Timeout waiting for next batch ({}s) - reader thread may be stalled",
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Channel closed - check error state first
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                // Then check if thread panicked
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

    /// Get mutable access to the output writer.
    #[allow(dead_code)]
    pub fn writer(&mut self) -> &mut OutputWriter {
        &mut self.writer
    }

    /// Flush the output and wait for the prefetch thread to complete.
    pub fn finish(&mut self) -> Result<()> {
        self.writer.finish()?;

        // Wait for the prefetch thread to complete
        if let Some(handle) = self.prefetch_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow!("Prefetch thread panicked"))?;
        }

        Ok(())
    }
}

// ============================================================================
// Threaded gzip decompression
// ============================================================================

/// Gzip magic bytes, as needletail detects them.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Bytes per decompressed chunk handed from a decompression thread to the parser.
const DECOMPRESS_CHUNK_SIZE: usize = 1 << 20;

/// Decompressed chunks queued per input file. With the chunk being filled and
/// the chunk being parsed, read-ahead is at most `DECOMPRESS_CHANNEL_DEPTH + 2`
/// chunks (about 6 MiB) per file.
const DECOMPRESS_CHANNEL_DEPTH: usize = 4;

/// A chunk sent by a decompression thread. An empty chunk marks end of input.
type Chunk = std::io::Result<Vec<u8>>;

/// Open a FASTX file for parsing, inflating gzip input on a dedicated thread.
///
/// Inflation is the slowest part of reading gzipped FASTQ. A thread per file
/// overlaps it with parsing on the reader thread and lets R1 and R2 inflate
/// concurrently instead of one after the other. Other input goes to needletail
/// unchanged: plain input has nothing to offload, and zstd (much faster to
/// decompress than gzip) has not been measured as a bottleneck.
fn open_fastx(path: &Path) -> Result<Box<dyn FastxReader>, ParseError> {
    open_fastx_with_chunk_size(path, DECOMPRESS_CHUNK_SIZE)
}

fn open_fastx_with_chunk_size(
    path: &Path,
    chunk_size: usize,
) -> Result<Box<dyn FastxReader>, ParseError> {
    let mut file = File::open(path)?;
    let mut magic = Vec::with_capacity(GZIP_MAGIC.len());
    (&mut file)
        .take(GZIP_MAGIC.len() as u64)
        .read_to_end(&mut magic)?;
    let is_gzip = magic == GZIP_MAGIC;
    // Put the peeked bytes back in front rather than seeking, so pipes still work.
    let input = Cursor::new(magic).chain(file);
    if !is_gzip {
        return parse_fastx_reader(input);
    }

    // MultiGzDecoder, like needletail, so multi-member files (bgzip, `cat a.gz
    // b.gz`) are read to the end. The thread is not joined: it exits after end
    // of input, on a decode error, or once the parser drops the receiver.
    let decoder = MultiGzDecoder::new(input);
    let (tx, rx) = mpsc::sync_channel(DECOMPRESS_CHANNEL_DEPTH);
    thread::Builder::new()
        .name("rype-gunzip".to_string())
        .spawn(move || send_chunks(decoder, chunk_size, tx))?;

    // Detect FASTA/FASTQ from the first decompressed byte, as needletail's own
    // gzip branch does. Passing the stream back through parse_fastx_reader
    // would sniff for compression a second time and report any read error at
    // the start as "empty file".
    let mut reader = ChunkReader::new(rx);
    let mut first = [0u8; 1];
    reader.read_exact(&mut first).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => ParseError::new_empty_file(),
        _ => e.into(),
    })?;
    let reader = Cursor::new(first).chain(reader);
    match first[0] {
        b'>' => Ok(Box::new(FastaReader::new(reader))),
        b'@' => Ok(Box::new(FastqReader::new(reader))),
        byte => Err(ParseError::new_unknown_format(byte)),
    }
}

/// Read `source` to the end in `chunk_size` pieces and send them, followed by an
/// empty end-of-input chunk. Stops early if the receiver has been dropped.
fn send_chunks(mut source: impl Read, chunk_size: usize, tx: SyncSender<Chunk>) {
    loop {
        let mut chunk = Vec::with_capacity(chunk_size);
        match (&mut source)
            .take(chunk_size as u64)
            .read_to_end(&mut chunk)
        {
            Ok(_) => {
                let end = chunk.is_empty();
                if tx.send(Ok(chunk)).is_err() || end {
                    return;
                }
            }
            Err(e) => {
                // Pass on what inflated before the error, as reading the decoder
                // directly would, so the same records reach the parser first.
                if !chunk.is_empty() && tx.send(Ok(chunk)).is_err() {
                    return;
                }
                let _ = tx.send(Err(e));
                return;
            }
        }
    }
}

/// `Read` over the chunks sent by [`send_chunks`].
struct ChunkReader {
    rx: Receiver<Chunk>,
    chunk: Cursor<Vec<u8>>,
    done: bool,
}

impl ChunkReader {
    fn new(rx: Receiver<Chunk>) -> Self {
        Self {
            rx,
            chunk: Cursor::new(Vec::new()),
            done: false,
        }
    }
}

impl Read for ChunkReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let n = self.chunk.read(buf)?;
            // n == 0 with a non-empty buf means the current chunk is used up.
            if n > 0 || self.done || buf.is_empty() {
                return Ok(n);
            }
            match self.rx.recv() {
                Ok(Ok(chunk)) => {
                    self.done = chunk.is_empty();
                    self.chunk = Cursor::new(chunk);
                }
                Ok(Err(e)) => return Err(e),
                // The sender dropped without the end-of-input chunk: the thread
                // died mid-file. Reporting EOF here would truncate input silently.
                Err(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "gzip decompression thread exited before the end of input",
                    ))
                }
            }
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use needletail::parse_fastx_file;

    // -------------------------------------------------------------------------
    // Tests for OwnedFastxRecord
    // -------------------------------------------------------------------------

    #[test]
    fn test_owned_fastx_record_creation_single_end_fasta() {
        let record = OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None);
        assert_eq!(record.query_id, 0);
        assert_eq!(record.seq1, b"ACGT");
        assert!(!record.is_fastq());
        assert!(!record.is_paired());
    }

    #[test]
    fn test_owned_fastx_record_creation_paired_fastq() {
        let record = OwnedFastxRecord::new(
            1,
            b"ACGT".to_vec(),
            Some(b"IIII".to_vec()),
            Some(b"TGCA".to_vec()),
            Some(b"JJJJ".to_vec()),
        );
        assert!(record.is_fastq());
        assert!(record.is_paired());
        assert_eq!(record.qual1.as_ref().unwrap(), b"IIII");
        assert_eq!(record.seq2.as_ref().unwrap(), b"TGCA");
        assert_eq!(record.qual2.as_ref().unwrap(), b"JJJJ");
    }

    #[test]
    fn test_owned_fastx_record_sequences_tuple() {
        let record = OwnedFastxRecord::new(5, b"AA".to_vec(), None, Some(b"TT".to_vec()), None);
        let (s1, s2) = record.sequences();
        assert_eq!(s1, b"AA");
        assert_eq!(s2.unwrap(), b"TT");
    }

    // -------------------------------------------------------------------------
    // Tests for base_read_id
    // -------------------------------------------------------------------------

    #[test]
    fn test_base_read_id_simple() {
        // Simple read ID with no suffix or comment
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_space_comment() {
        // Read ID with space-separated comment
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1 comment text here"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_with_tab_comment() {
        // Read ID with tab-separated comment
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1\tcomment"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_with_slash_1_suffix() {
        // Read ID with /1 suffix (forward read)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/1"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_slash_2_suffix() {
        // Read ID with /2 suffix (reverse read)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/2"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_slash_1_and_comment() {
        // Read ID with /1 suffix followed by comment
        // The /1 should be stripped, then the space would stop parsing
        // But since /1 is at position 5, and space is at position 7,
        // we find space first... no wait, we iterate and check both conditions.
        // Actually the logic finds first space OR /1 at end of ID portion.
        // Let's trace: "READ1/1 comment"
        // - First find space at position 7
        // - id_portion = "READ1/1"
        // - Check if ends with /1 or /2: yes, /1
        // - Return "READ1"
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1/1 comment"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_illumina_style() {
        // Illumina-style header: @HWUSI-EAS100R:6:73:941:1973/1 length=36
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"HWUSI-EAS100R:6:73:941:1973/1 length=36"),
            b"HWUSI-EAS100R:6:73:941:1973"
        );
    }

    #[test]
    fn test_base_read_id_casava_style() {
        // CASAVA 1.8+ style: @EAS139:136:FC706VJ:2:2104:15343:197393 1:Y:18:ATCACG
        assert_eq!(
            PrefetchingIoHandler::base_read_id(
                b"EAS139:136:FC706VJ:2:2104:15343:197393 1:Y:18:ATCACG"
            ),
            b"EAS139:136:FC706VJ:2:2104:15343:197393"
        );
    }

    #[test]
    fn test_base_read_id_empty() {
        // Empty input
        assert_eq!(PrefetchingIoHandler::base_read_id(b""), b"");
    }

    #[test]
    fn test_base_read_id_slash_only() {
        // Just a slash (edge case)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"/"), b"/");
    }

    #[test]
    fn test_base_read_id_slash_3() {
        // /3 should NOT be stripped (only /1 and /2 are paired-end markers)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/3"), b"READ1/3");
    }

    #[test]
    fn test_base_read_id_internal_slash() {
        // Internal slash should not affect anything
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"path/to/READ1"),
            b"path/to/READ1"
        );
    }

    #[test]
    fn test_base_read_id_multiple_spaces() {
        // Multiple spaces - only first one matters
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1  extra  spaces"),
            b"READ1"
        );
    }

    // -------------------------------------------------------------------------
    // Tests for preserve_for_output (quality score preservation)
    // -------------------------------------------------------------------------

    #[test]
    fn test_reader_preserves_quality_when_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "@read1\nACGT\n+\nIIII").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = true
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            None,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(records[0].is_fastq());
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
    }

    #[test]
    fn test_reader_skips_quality_when_not_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "@read1\nACGT\n+\nIIII").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = false (default behavior)
        let mut handler =
            PrefetchingIoHandler::with_trim(tmp.path(), None, None, 100, None).unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(!records[0].is_fastq()); // qual1 is None
        assert!(records[0].qual1.is_none());
    }

    #[test]
    fn test_reader_paired_quality_when_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create paired FASTQ files
        let mut tmp_r1 = NamedTempFile::new().unwrap();
        writeln!(tmp_r1, "@read1/1\nACGT\n+\nIIII").unwrap();
        tmp_r1.flush().unwrap();

        let mut tmp_r2 = NamedTempFile::new().unwrap();
        writeln!(tmp_r2, "@read1/2\nTGCA\n+\nJJJJ").unwrap();
        tmp_r2.flush().unwrap();

        let r2_path = tmp_r2.path().to_path_buf();

        // preserve_for_output = true
        let mut handler = PrefetchingIoHandler::with_options(
            tmp_r1.path(),
            Some(&r2_path),
            None,
            100,
            None,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(records[0].is_fastq());
        assert!(records[0].is_paired());
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
        assert_eq!(records[0].qual2.as_ref().unwrap(), b"JJJJ");
        assert_eq!(records[0].seq2.as_ref().unwrap(), b"TGCA");
    }

    #[test]
    fn test_reader_fasta_has_no_quality() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, ">read1\nACGT").unwrap();
        tmp.flush().unwrap();

        // Even with preserve_for_output = true, FASTA has no quality
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            None,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(!records[0].is_fastq());
        assert!(records[0].qual1.is_none());
    }

    #[test]
    fn test_reader_quality_trimmed_with_sequence() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        // 8bp sequence and quality
        writeln!(tmp, "@read1\nACGTACGT\n+\nIIIIJJJJ").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = true with trim_to = 4
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            Some(4), // trim_to
            None,    // minimum_length
            true,    // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        // Sequence should be trimmed to 4bp
        assert_eq!(records[0].seq1, b"ACGT");
        // Quality should also be trimmed to 4bp
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
    }

    // -------------------------------------------------------------------------
    // Tests for minimum_length filtering
    // -------------------------------------------------------------------------

    #[test]
    fn test_fastx_reader_minimum_length_filters_short_reads() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // 3 reads: 30bp, 80bp, 50bp
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, ">read1\n{}", "A".repeat(30)).unwrap();
        writeln!(tmp, ">read2\n{}", "C".repeat(80)).unwrap();
        writeln!(tmp, ">read3\n{}", "G".repeat(50)).unwrap();
        tmp.flush().unwrap();

        // minimum_length=50 should skip read1 (30bp), keep read2 (80bp) and read3 (50bp)
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            None,     // trim_to
            Some(50), // minimum_length
            false,
        )
        .unwrap();

        let (records, headers) = handler.next_batch().unwrap().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(headers, vec!["read2", "read3"]);
        assert_eq!(records[0].seq1.len(), 80);
        assert_eq!(records[1].seq1.len(), 50);
    }

    #[test]
    fn test_fastx_reader_minimum_length_before_trim_to() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // 3 reads: 40bp, 100bp, 60bp
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, ">read1\n{}", "A".repeat(40)).unwrap();
        writeln!(tmp, ">read2\n{}", "C".repeat(100)).unwrap();
        writeln!(tmp, ">read3\n{}", "G".repeat(60)).unwrap();
        tmp.flush().unwrap();

        // minimum_length=50, trim_to=70
        // read1 (40bp): skipped by minimum_length
        // read2 (100bp): passes min_len, trimmed to 70
        // read3 (60bp): passes min_len, passes trim_to (60 < 70 → skipped by trim_to)
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            Some(70), // trim_to
            Some(50), // minimum_length
            false,
        )
        .unwrap();

        let (records, headers) = handler.next_batch().unwrap().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(headers, vec!["read2"]);
        assert_eq!(records[0].seq1.len(), 70); // trimmed from 100 to 70
    }

    #[test]
    fn test_fastx_reader_minimum_length_paired_end() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // R1: 40bp and 60bp
        let mut tmp_r1 = NamedTempFile::new().unwrap();
        writeln!(tmp_r1, ">read1/1\n{}", "A".repeat(40)).unwrap();
        writeln!(tmp_r1, ">read2/1\n{}", "C".repeat(60)).unwrap();
        tmp_r1.flush().unwrap();

        // R2: 50bp and 70bp
        let mut tmp_r2 = NamedTempFile::new().unwrap();
        writeln!(tmp_r2, ">read1/2\n{}", "T".repeat(50)).unwrap();
        writeln!(tmp_r2, ">read2/2\n{}", "G".repeat(70)).unwrap();
        tmp_r2.flush().unwrap();

        let r2_path = tmp_r2.path().to_path_buf();

        // minimum_length=50: read1 R1=40bp skipped (with its R2), read2 R1=60bp kept
        let mut handler = PrefetchingIoHandler::with_options(
            tmp_r1.path(),
            Some(&r2_path),
            None,
            100,
            None,     // trim_to
            Some(50), // minimum_length
            false,
        )
        .unwrap();

        let (records, headers) = handler.next_batch().unwrap().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(headers, vec!["read2"]);
        assert_eq!(records[0].seq1.len(), 60);
        assert_eq!(records[0].seq2.as_ref().unwrap().len(), 70);
    }

    #[test]
    fn test_fastx_reader_minimum_length_gt_trim_to() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // 3 reads: 80bp, 120bp, 100bp
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, ">read1\n{}", "A".repeat(80)).unwrap();
        writeln!(tmp, ">read2\n{}", "C".repeat(120)).unwrap();
        writeln!(tmp, ">read3\n{}", "G".repeat(100)).unwrap();
        tmp.flush().unwrap();

        // minimum_length=100, trim_to=50
        // minimum_length is binding: reads must be >= 100bp
        // read1 (80bp): skipped by minimum_length
        // read2 (120bp): passes min_len (>= 100), trimmed to 50
        // read3 (100bp): passes min_len (>= 100), trimmed to 50
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            Some(50),  // trim_to
            Some(100), // minimum_length
            false,
        )
        .unwrap();

        let (records, headers) = handler.next_batch().unwrap().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(headers, vec!["read2", "read3"]);
        // Both surviving reads are trimmed to 50
        assert_eq!(records[0].seq1.len(), 50);
        assert_eq!(records[1].seq1.len(), 50);
    }

    // -------------------------------------------------------------------------
    // Tests for threaded gzip decompression
    // -------------------------------------------------------------------------

    /// FASTQ text whose records differ in id, length, sequence and quality, so a
    /// dropped, duplicated or misjoined record changes what is parsed.
    fn sample_fastq(n: usize, mate: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..n {
            let len = 20 + (i * 7) % 61;
            let seq: Vec<u8> = (0..len).map(|j| b"ACGT"[(i * 31 + j * 17) % 4]).collect();
            let qual: Vec<u8> = (0..len).map(|j| b'!' + ((i + j) % 40) as u8).collect();
            out.extend_from_slice(format!("@read{}/{} c{}\n", i, mate, i).as_bytes());
            out.extend_from_slice(&seq);
            out.extend_from_slice(b"\n+\n");
            out.extend_from_slice(&qual);
            out.push(b'\n');
        }
        out
    }

    /// Gzip each part as a separate member and concatenate them, as `bgzip` and
    /// `cat a.gz b.gz` produce.
    fn gzip_members(parts: &[&[u8]]) -> Vec<u8> {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        let mut out = Vec::new();
        for part in parts {
            let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
            enc.write_all(part).unwrap();
            out.extend(enc.finish().unwrap());
        }
        out
    }

    fn temp_file(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut tmp, bytes).unwrap();
        tmp
    }

    type ParsedRecord = (Vec<u8>, Vec<u8>, Option<Vec<u8>>);

    /// Records parsed before the first error (from opening or reading), and
    /// that error's message.
    fn parse_outcome(
        reader: Result<Box<dyn FastxReader>, ParseError>,
    ) -> (Vec<ParsedRecord>, Option<String>) {
        let mut out = Vec::new();
        let mut reader = match reader {
            Ok(reader) => reader,
            Err(e) => return (out, Some(e.to_string())),
        };
        while let Some(rec) = reader.next() {
            match rec {
                Ok(rec) => out.push((
                    rec.id().to_vec(),
                    rec.seq().to_vec(),
                    rec.qual().map(|q| q.to_vec()),
                )),
                Err(e) => return (out, Some(e.to_string())),
            }
        }
        (out, None)
    }

    #[test]
    fn test_threaded_gzip_parses_same_records_as_needletail() {
        // The decompression thread hands the parser fixed-size chunks, so record
        // boundaries fall mid-line; chunk size 1 puts a boundary between every
        // byte. Two gzip members check that inflation continues past the first
        // member, as needletail's own MultiGzDecoder does: stopping there would
        // silently drop the reads in every later member.
        let n = 500;
        let plain = sample_fastq(n, 1);
        let (head, tail) = plain.split_at(plain.len() / 3);
        let tmp = temp_file(&gzip_members(&[head, tail]));

        let expected = parse_outcome(parse_fastx_file(tmp.path()));
        assert_eq!(
            (expected.0.len(), &expected.1),
            (n, &None),
            "precondition: needletail reads every member"
        );

        for chunk_size in [1, 7, 4096, DECOMPRESS_CHUNK_SIZE] {
            assert_eq!(
                parse_outcome(open_fastx_with_chunk_size(tmp.path(), chunk_size)),
                expected,
                "chunk_size {}",
                chunk_size
            );
        }
    }

    #[test]
    fn test_threaded_gzip_fails_like_needletail_on_bad_input() {
        // Bad input must fail the same way it did before threading: the same
        // records reach the caller before the error, and the error names the
        // real problem. Small files keep each case inside the first chunk,
        // where a decode error could otherwise surface as "empty file".
        let plain = sample_fastq(300, 1);
        let gz = gzip_members(&[&plain]);
        let mut bad_crc = gz.clone();
        let crc_byte = bad_crc.len() - 8;
        bad_crc[crc_byte] ^= 0xff;
        let cases: [(&str, Vec<u8>); 4] = [
            // Every record inflates; only the decoder's checksum error reveals
            // the corruption, so this fails only if that error is passed on.
            ("bad checksum", bad_crc),
            ("truncated", gz[..gz.len() / 2].to_vec()),
            // Decompressing once leaves gzip bytes, not FASTQ: an unknown format.
            ("double gzip", gzip_members(&[&gz])),
            ("empty", gzip_members(&[b""])),
        ];

        for (name, bytes) in cases {
            let tmp = temp_file(&bytes);
            let expected = parse_outcome(parse_fastx_file(tmp.path()));
            assert!(
                expected.1.is_some(),
                "precondition: needletail rejects {}",
                name
            );
            for chunk_size in [7, DECOMPRESS_CHUNK_SIZE] {
                assert_eq!(
                    parse_outcome(open_fastx_with_chunk_size(tmp.path(), chunk_size)),
                    expected,
                    "{} (chunk_size {})",
                    name,
                    chunk_size
                );
            }
        }
    }

    #[test]
    fn test_chunk_reader_errors_when_thread_exits_without_end_marker() {
        // If the decompression thread dies (e.g. panics) its sender drops without
        // the empty end-of-input chunk. That must read as an error, not as EOF,
        // or the input would be truncated silently.
        let (tx, rx) = mpsc::sync_channel(4);
        tx.send(Ok(b"@r\nACGT\n".to_vec())).unwrap();
        drop(tx);
        let mut buf = Vec::new();
        assert!(ChunkReader::new(rx).read_to_end(&mut buf).is_err());

        let (tx, rx) = mpsc::sync_channel(4);
        tx.send(Ok(b"@r\nACGT\n".to_vec())).unwrap();
        tx.send(Ok(Vec::new())).unwrap();
        drop(tx);
        let mut reader = ChunkReader::new(rx);
        // A zero-length read must not consume (and so lose) the pending chunk.
        assert_eq!(reader.read(&mut []).unwrap(), 0);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"@r\nACGT\n");
    }

    #[test]
    fn test_paired_gzip_input_matches_plain_input() {
        // End to end through the prefetching reader: paired gzip input (each
        // file inflated on its own thread) must produce exactly the batches that
        // the same reads produce uncompressed. A small batch size makes the
        // pairing cross many batches.
        let n = 1000;
        let (p1, p2) = (sample_fastq(n, 1), sample_fastq(n, 2));
        let (plain1, plain2) = (temp_file(&p1), temp_file(&p2));
        let (gz1, gz2) = (
            temp_file(&gzip_members(&[&p1])),
            temp_file(&gzip_members(&[&p2])),
        );

        let read_batches = |r1: &Path, r2: &Path| {
            let r2 = r2.to_path_buf();
            let mut handler =
                PrefetchingIoHandler::with_options(r1, Some(&r2), None, 64, None, None, true)
                    .unwrap();
            let mut out = Vec::new();
            while let Some((records, headers)) = handler.next_batch().unwrap() {
                for (rec, header) in records.into_iter().zip(headers) {
                    out.push((
                        header,
                        rec.query_id,
                        rec.seq1,
                        rec.qual1,
                        rec.seq2,
                        rec.qual2,
                    ));
                }
            }
            handler.finish().unwrap();
            out
        };

        let expected = read_batches(plain1.path(), plain2.path());
        assert_eq!(expected.len(), n);
        assert_eq!(read_batches(gz1.path(), gz2.path()), expected);
    }
}
