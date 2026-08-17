//! Tests for `rype_classify_arrow_ex` / `rype_classify_arrow_log_ratio_ex`,
//! which decouple the input RecordBatch size from the classification batch size.
//!
//! The reason those entry points exist is that classification cost is dominated
//! by the pass over the index's Parquet shards, and the original Arrow entry
//! points run that pass once per input batch. A caller that wants to bound the
//! sequence bytes held in any one input batch — and so bound peak memory
//! independently of corpus size — could only do it by paying for extra index
//! passes.
//!
//! Two things therefore have to hold, and both are asserted here:
//!
//! 1. **Results do not depend on how the input was split into batches.** If they
//!    did, a caller could not choose an input batch size for memory reasons.
//! 2. **The number of classification passes tracks `classify_batch_rows`, not
//!    the input batch count.** One output batch is emitted per pass, so the
//!    output batch count is the observable proxy. Without this the entry point
//!    would be pointless — it would trade memory for I/O.

#![cfg(feature = "arrow-ffi")]

use anyhow::Result;
use arrow::array::{
    Array, BinaryArray, Float64Array, Int64Array, RecordBatchIterator, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use std::ffi::CString;
use std::sync::Arc;
use tempfile::tempdir;

use rype::c_api::{
    rype_classify_arrow, rype_classify_arrow_best_hit, rype_classify_arrow_best_hit_ex,
    rype_classify_arrow_ex, rype_classify_arrow_log_ratio, rype_classify_arrow_log_ratio_ex,
    rype_get_last_error, rype_index_free, rype_index_load, rype_negative_set_create,
    rype_negative_set_free, RypeIndex, RypeNegativeSet,
};
use rype::{
    create_parquet_inverted_index, extract_into, BucketData, MinimizerWorkspace,
    ParquetWriteOptions,
};

const K: usize = 16;
const W: usize = 5;
const SALT: u64 = 0x12345;

mod common;
use common::generate_sequence;

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("sequence", DataType::Binary, false),
    ]))
}

fn make_batch(ids: &[i64], seqs: &[Vec<u8>]) -> RecordBatch {
    let seq_refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(BinaryArray::from_iter_values(seq_refs)),
        ],
    )
    .unwrap()
}

/// Split `count` reads into batches of `per_batch` rows.
fn make_input_batches(count: usize, per_batch: usize) -> Vec<RecordBatch> {
    let mut batches = Vec::new();
    let mut next = 0usize;
    while next < count {
        let end = (next + per_batch).min(count);
        let ids: Vec<i64> = (next..end).map(|i| i as i64).collect();
        let seqs: Vec<Vec<u8>> = (next..end)
            .map(|i| generate_sequence(150, i as u64))
            .collect();
        batches.push(make_batch(&ids, &seqs));
        next = end;
    }
    batches
}

/// Batches whose reads alternate between slices of the bucket-`seed` reference
/// (every minimizer present, so numerator score 1.0) and novel sequences (score
/// ~0). A skip threshold between the two therefore splits the batch, which is
/// what makes the log-ratio fast path observable — the shared
/// `make_input_batches` fixture scores all-or-nothing and cannot produce a mix.
fn make_mixed_input_batches(count: usize, per_batch: usize, seed: u64) -> Vec<RecordBatch> {
    let reference = generate_sequence(4000, seed);
    let mut batches = Vec::new();
    let mut next = 0usize;
    while next < count {
        let end = (next + per_batch).min(count);
        let ids: Vec<i64> = (next..end).map(|i| i as i64).collect();
        let seqs: Vec<Vec<u8>> = (next..end)
            .map(|i| {
                if i % 2 == 0 {
                    let off = (i * 97) % (reference.len() - 150);
                    reference[off..off + 150].to_vec()
                } else {
                    generate_sequence(150, 900_000 + i as u64)
                }
            })
            .collect();
        batches.push(make_batch(&ids, &seqs));
        next = end;
    }
    batches
}

fn ffi_input_stream(batches: Vec<RecordBatch>) -> FFI_ArrowArrayStream {
    let schema = input_schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    FFI_ArrowArrayStream::new(Box::new(reader))
}

/// Release a stream the callee never took ownership of.
///
/// rype.h: on -1 from argument validation the stream was never consumed and the
/// caller still owns it. Tests that exercise those rejections must release it or
/// they leak the boxed reader and its batches.
fn release_unconsumed(mut stream: FFI_ArrowArrayStream) {
    if let Some(release) = stream.release {
        unsafe { release(&mut stream) };
    }
}

fn last_error() -> String {
    let ptr = rype_get_last_error();
    if ptr.is_null() {
        return "(no error)".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// One classification result row: (query_id, bucket_id, score).
type HitRow = (i64, u32, f64);
/// One log-ratio result row: (query_id, log_ratio, fast_path).
type LogRatioRow = (i64, f64, i32);

/// Drain an output stream into (batch count, rows sorted for comparison).
fn drain_hits(out: FFI_ArrowArrayStream) -> Result<(usize, Vec<HitRow>)> {
    let mut out = out;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut out) }?;
    let mut batch_count = 0usize;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch?;
        batch_count += 1;
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let buckets = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), buckets.value(i), scores.value(i)));
        }
    }
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((batch_count, rows))
}

/// Drain a log-ratio output stream into (batch count, rows sorted by query id).
fn drain_log_ratios(out: FFI_ArrowArrayStream) -> Result<(usize, Vec<LogRatioRow>)> {
    let mut out = out;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut out) }?;
    let mut batch_count = 0usize;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch?;
        batch_count += 1;
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ratios = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let fast = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), ratios.value(i), fast.value(i)));
        }
    }
    // Sort on the query id alone. Ratios may be NaN (compute_log_ratio returns
    // NaN for a read matching neither index), and a tuple comparison reaching
    // one makes partial_cmp return None, panicking the unwrap.
    rows.sort_by_key(|r| r.0);
    Ok((batch_count, rows))
}

/// Compare log-ratio results treating NaN as equal to NaN.
///
/// A read matching neither index yields NaN by design (compute_log_ratio, and
/// rype.h lists NaN as a valid log_ratio). Under `assert_eq!` such a row never
/// compares equal to itself, so a test asserting "batching did not change the
/// result" would fail for a reason that has nothing to do with batching.
fn log_ratios_match(a: &[LogRatioRow], b: &[LogRatioRow]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.0 == y.0 && x.2 == y.2 && (x.1 == y.1 || (x.1.is_nan() && y.1.is_nan()))
        })
}

/// Build a Parquet index whose buckets are drawn from the same generator as the
/// queries, so classification produces real hits rather than an empty result.
fn build_index(dir: &std::path::Path, name: &str, seeds: &[u64]) -> Result<std::path::PathBuf> {
    let index_path = dir.join(name);
    let mut ws = MinimizerWorkspace::new();
    let mut buckets = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        extract_into(&generate_sequence(4000, *seed), K, W, SALT, &mut ws);
        let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
        mins.sort_unstable();
        mins.dedup();
        buckets.push(BucketData {
            bucket_id: (i + 1) as u32,
            bucket_name: format!("bucket{}", i + 1),
            sources: vec![format!("src{}", i + 1)],
            minimizers: mins,
        });
    }
    create_parquet_inverted_index(
        &index_path,
        buckets,
        K,
        W,
        SALT,
        None,
        Some(&ParquetWriteOptions::default()),
        None,
    )?;
    Ok(index_path)
}

fn load(path: &std::path::Path) -> *mut RypeIndex {
    let cstr = CString::new(path.to_str().unwrap()).unwrap();
    let ptr = rype_index_load(cstr.as_ptr());
    assert!(!ptr.is_null(), "index load failed: {}", last_error());
    ptr
}

#[test]
fn classify_arrow_ex_results_are_independent_of_input_batching() -> Result<()> {
    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7, 11])?;
    let index = load(&index_path);

    // Baseline: the pre-existing entry point, whole corpus in one input batch.
    let mut input = ffi_input_stream(make_input_batches(64, 64));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow(index, std::ptr::null(), &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow failed: {}", last_error());
    let (baseline_batches, baseline) = drain_hits(out)?;
    assert_eq!(baseline_batches, 1);
    assert!(
        !baseline.is_empty(),
        "fixture produced no hits — the test would assert nothing"
    );

    // Same reads, 16 input batches, one classification pass over all of them.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 64, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (accumulated_batches, accumulated) = drain_hits(out)?;

    assert_eq!(
        accumulated, baseline,
        "splitting the input into batches changed the results"
    );
    assert_eq!(
        accumulated_batches, 1,
        "16 input batches with classify_batch_rows=64 must still be one \
         classification pass — otherwise the entry point buys memory with I/O"
    );

    // classify_batch_rows=16 over 16 input batches of 4: a pass every 4 batches.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 16, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (grouped_batches, grouped) = drain_hits(out)?;
    assert_eq!(grouped, baseline, "grouping changed the results");
    assert_eq!(
        grouped_batches, 4,
        "expected 64/16 = 4 classification passes"
    );

    // classify_batch_rows=0 must reproduce the old one-pass-per-input-batch shape.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 0, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (per_batch_batches, per_batch) = drain_hits(out)?;
    assert_eq!(per_batch, baseline);
    assert_eq!(
        per_batch_batches, 16,
        "classify_batch_rows=0 must classify each input batch on its own"
    );

    rype_index_free(index);
    Ok(())
}

#[test]
fn classify_arrow_ex_negative_filtering_is_independent_of_input_batching() -> Result<()> {
    let dir = tempdir()?;
    let pos_path = build_index(dir.path(), "pos.ryxdi", &[7, 11])?;
    // The negative index shares seed 7 with the positive one, so it actually
    // removes minimizers rather than being a no-op.
    let neg_path = build_index(dir.path(), "neg.ryxdi", &[7])?;

    let index = load(&pos_path);
    let neg_index = load(&neg_path);
    let neg_set: *mut RypeNegativeSet = rype_negative_set_create(neg_index);
    assert!(!neg_set.is_null(), "negative set: {}", last_error());

    let mut input = ffi_input_stream(make_input_batches(48, 48));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow(index, neg_set, &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow failed: {}", last_error());
    let (_, baseline) = drain_hits(out)?;

    let mut input = ffi_input_stream(make_input_batches(48, 6));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow_ex(index, neg_set, &mut input, 0.0, 48, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (accumulated_batches, accumulated) = drain_hits(out)?;

    assert_eq!(
        accumulated, baseline,
        "negative filtering must not depend on input batching"
    );
    assert_eq!(accumulated_batches, 1);

    // Both runs above pass a negative set, so if filtering silently became a
    // no-op — an empty hitting set, or the retain loop in classify_arrow_internal
    // deleted — they would shift together and still compare equal. Pin that the
    // negative set actually changes the outcome.
    let mut input = ffi_input_stream(make_input_batches(48, 6));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 48, &mut out) };
    assert_eq!(rc, 0, "unfiltered run failed: {}", last_error());
    let (_, unfiltered) = drain_hits(out)?;
    assert!(
        !unfiltered.is_empty(),
        "fixture produced no hits — nothing to filter"
    );
    assert_ne!(
        accumulated, unfiltered,
        "the negative index shares seed 7 with the positive one, so filtering \
         must change the results; identical output means it did nothing"
    );

    rype_negative_set_free(neg_set);
    rype_index_free(neg_index);
    rype_index_free(index);
    Ok(())
}

#[test]
fn classify_arrow_log_ratio_ex_results_are_independent_of_input_batching() -> Result<()> {
    let dir = tempdir()?;
    // Log-ratio requires single-bucket indices.
    let num_path = build_index(dir.path(), "num.ryxdi", &[7])?;
    let denom_path = build_index(dir.path(), "denom.ryxdi", &[11])?;

    let num = load(&num_path);
    let denom = load(&denom_path);

    let mut input = ffi_input_stream(make_input_batches(32, 32));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow_log_ratio(num, denom, &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0, "log_ratio failed: {}", last_error());
    let (_, baseline) = drain_log_ratios(out)?;
    assert_eq!(baseline.len(), 32, "one result per read");

    let mut input = ffi_input_stream(make_input_batches(32, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow_log_ratio_ex(num, denom, &mut input, 0.0, 32, &mut out) };
    assert_eq!(rc, 0, "log_ratio_ex failed: {}", last_error());
    let (accumulated_batches, accumulated) = drain_log_ratios(out)?;

    assert!(
        log_ratios_match(&accumulated, &baseline),
        "splitting the input into batches changed the log ratios"
    );
    assert_eq!(accumulated_batches, 1);

    rype_index_free(denom);
    rype_index_free(num);
    Ok(())
}

/// The reader must not lose the tail when the last group is short, and must not
/// emit a spurious empty batch when the input divides evenly.
#[test]
fn classify_arrow_ex_handles_partial_and_exact_final_groups() -> Result<()> {
    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7])?;
    let index = load(&index_path);

    // 10 reads, groups of 4 → 4 + 4 + 2.
    let mut input = ffi_input_stream(make_input_batches(10, 1));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 4, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (partial_batches, partial) = drain_hits(out)?;
    assert_eq!(partial_batches, 3, "4 + 4 + 2");

    // 8 reads, groups of 4 → exactly 2, with no trailing empty batch.
    let mut input = ffi_input_stream(make_input_batches(8, 1));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 4, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (exact_batches, _) = drain_hits(out)?;
    assert_eq!(exact_batches, 2, "no trailing empty batch");

    // The 10-read run must contain every read the 10 individual reads produce.
    let mut input = ffi_input_stream(make_input_batches(10, 10));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow(index, std::ptr::null(), &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0);
    let (_, baseline) = drain_hits(out)?;
    assert_eq!(partial, baseline, "the short final group lost reads");

    rype_index_free(index);
    Ok(())
}

/// An input stream that yields nothing must produce an output stream that
/// yields nothing, not hang or emit an empty batch forever.
#[test]
fn classify_arrow_ex_handles_empty_input() -> Result<()> {
    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7])?;
    let index = load(&index_path);

    let mut input = ffi_input_stream(Vec::new());
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 64, &mut out) };
    assert_eq!(rc, 0, "rype_classify_arrow_ex failed: {}", last_error());
    let (batches, rows) = drain_hits(out)?;
    assert_eq!(batches, 0);
    assert!(rows.is_empty());

    rype_index_free(index);
    Ok(())
}

/// The output stream must advertise the classification result schema even
/// before any batch is pulled — consumers bind on it first.
#[test]
fn classify_arrow_ex_exposes_result_schema() -> Result<()> {
    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7])?;
    let index = load(&index_path);

    let mut input = ffi_input_stream(make_input_batches(4, 2));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_ex(index, std::ptr::null(), &mut input, 0.0, 64, &mut out) };
    assert_eq!(rc, 0);
    {
        let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut out) }?;
        assert_eq!(reader.schema(), rype::arrow::result_schema());
    }
    // The reader owns a pointer to the index; freeing it first is the
    // use-after-free the header's safety contract warns about, even though
    // dropping the reader happens not to dereference it today.
    rype_index_free(index);
    Ok(())
}

/// `classify_batch_rows` becomes one `QueryInvertedIndex`, which packs the read
/// index into 31 bits and asserts on the limit.
///
/// A caller passing SIZE_MAX to mean "no limit" must get a named error at the
/// entry point. Without this check the group would simply keep growing — an OOM
/// long before the limit, and past it an assert firing inside a rayon worker,
/// unwinding toward the C `get_next` callback.
#[test]
fn classify_arrow_ex_rejects_classify_batch_rows_above_the_read_limit() -> Result<()> {
    // Mirrors MAX_READS in src/constants.rs (31 bits; bit 31 is the strand flag).
    const MAX_READS: usize = 0x7FFF_FFFF;

    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7])?;
    let num_path = build_index(dir.path(), "num.ryxdi", &[7])?;
    let denom_path = build_index(dir.path(), "denom.ryxdi", &[11])?;
    let index = load(&index_path);
    let num = load(&num_path);
    let denom = load(&denom_path);

    let mut input = ffi_input_stream(make_input_batches(4, 2));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_ex(
            index,
            std::ptr::null(),
            &mut input,
            0.0,
            usize::MAX,
            &mut out,
        )
    };
    assert_eq!(rc, -1, "SIZE_MAX must be rejected, not accumulated");
    assert!(
        last_error().contains("classify_batch_rows"),
        "the error must name the offending parameter, got: {}",
        last_error()
    );
    release_unconsumed(input);

    // Same guard on the log-ratio entry point.
    let mut input = ffi_input_stream(make_input_batches(4, 2));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_log_ratio_ex(num, denom, &mut input, 0.0, usize::MAX, &mut out)
    };
    assert_eq!(rc, -1, "log-ratio must reject SIZE_MAX too");
    assert!(last_error().contains("classify_batch_rows"));
    release_unconsumed(input);

    // The limit itself is legal — the group just ends when the input does.
    let mut input = ffi_input_stream(make_input_batches(4, 2));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_ex(
            index,
            std::ptr::null(),
            &mut input,
            0.0,
            MAX_READS,
            &mut out,
        )
    };
    assert_eq!(rc, 0, "MAX_READS must be accepted: {}", last_error());
    let (batches, rows) = drain_hits(out)?;
    assert_eq!(batches, 1, "one group, ended by end-of-input");
    assert!(!rows.is_empty());

    rype_index_free(denom);
    rype_index_free(num);
    rype_index_free(index);
    Ok(())
}

/// A NULL index must be refused rather than dereferenced. The entry point also
/// rejects misaligned pointers, matching `rype_classify_arrow_log_ratio_ex` and
/// the other 20-odd entry points in c_api.rs; that half is not exercised here
/// because constructing a misaligned pointer would be undefined behaviour if the
/// guard were ever removed, and no test in this repo does it.
#[test]
fn classify_arrow_ex_rejects_null_index() -> Result<()> {
    let mut input = ffi_input_stream(make_input_batches(4, 2));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_ex(
            std::ptr::null(),
            std::ptr::null(),
            &mut input,
            0.0,
            64,
            &mut out,
        )
    };
    assert_eq!(rc, -1);
    assert!(
        last_error().contains("index"),
        "error should name the index pointer, got: {}",
        last_error()
    );
    release_unconsumed(input);
    Ok(())
}

/// `rype_classify_arrow_best_hit_ex` is the one `_ex` variant whose reduction is
/// applied per classification group rather than per input batch, so widening the
/// group widens what `filter_best_hits` sees. Query ids are unique per read here,
/// as the header requires, and under that precondition grouping must not change
/// which hit wins.
#[test]
fn classify_arrow_best_hit_ex_results_are_independent_of_input_batching() -> Result<()> {
    let dir = tempdir()?;
    let index_path = build_index(dir.path(), "pos.ryxdi", &[7, 11])?;
    let index = load(&index_path);

    // Baseline: the pre-existing best-hit entry point, one input batch.
    let mut input = ffi_input_stream(make_input_batches(64, 64));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_best_hit(index, std::ptr::null(), &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0, "best_hit failed: {}", last_error());
    let (baseline_batches, baseline) = drain_hits(out)?;
    assert_eq!(baseline_batches, 1);
    assert!(!baseline.is_empty(), "fixture produced no hits");

    // Best-hit must actually reduce: without this the test would pass even if
    // filter_best_hits were dropped from the pipeline entirely.
    let mut input = ffi_input_stream(make_input_batches(64, 64));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow(index, std::ptr::null(), &mut input, 0.0, &mut out) };
    assert_eq!(rc, 0);
    let (_, all_hits) = drain_hits(out)?;
    assert!(
        baseline.len() < all_hits.len(),
        "best-hit kept {} of {} rows — it is not filtering",
        baseline.len(),
        all_hits.len()
    );
    // One row per query at most is the defining property.
    let mut ids: Vec<i64> = baseline.iter().map(|r| r.0).collect();
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total, "best-hit emitted a query id twice");

    // 16 input batches, one classification group: one pass, same winners.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_best_hit_ex(index, std::ptr::null(), &mut input, 0.0, 64, &mut out)
    };
    assert_eq!(rc, 0, "best_hit_ex failed: {}", last_error());
    let (accumulated_batches, accumulated) = drain_hits(out)?;
    assert_eq!(
        accumulated, baseline,
        "grouping changed which hit won per query"
    );
    assert_eq!(
        accumulated_batches, 1,
        "expected a single classification pass"
    );

    // Four groups of 16: still one winner per query, unchanged.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_best_hit_ex(index, std::ptr::null(), &mut input, 0.0, 16, &mut out)
    };
    assert_eq!(rc, 0, "best_hit_ex failed: {}", last_error());
    let (grouped_batches, grouped) = drain_hits(out)?;
    assert_eq!(grouped, baseline, "regrouping changed the best hits");
    assert_eq!(
        grouped_batches, 4,
        "expected 64/16 = 4 classification passes"
    );

    // classify_batch_rows=0 reproduces one pass per input batch.
    let mut input = ffi_input_stream(make_input_batches(64, 4));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_best_hit_ex(index, std::ptr::null(), &mut input, 0.0, 0, &mut out)
    };
    assert_eq!(rc, 0, "best_hit_ex failed: {}", last_error());
    let (per_batch_batches, _) = drain_hits(out)?;
    assert_eq!(per_batch_batches, 16);

    rype_index_free(index);
    Ok(())
}

/// The log-ratio fast path partitions each group into "numerator score is high
/// enough to skip the denominator" and "needs the denominator". That partition is
/// computed per group, so it is the part most exposed to a change in group size —
/// and with `numerator_skip_threshold = 0.0` (as the other log-ratio test uses)
/// it never runs at all, because the threshold is then disabled.
#[test]
fn classify_arrow_log_ratio_ex_fast_path_split_is_independent_of_input_batching() -> Result<()> {
    const FAST_NONE: i32 = 0;
    const FAST_NUM_HIGH: i32 = 1;

    let dir = tempdir()?;
    let num_path = build_index(dir.path(), "num.ryxdi", &[7])?;
    let denom_path = build_index(dir.path(), "denom.ryxdi", &[11])?;
    let num = load(&num_path);
    let denom = load(&denom_path);

    // Half the reads are slices of the numerator reference (score 1.0), half are
    // novel (score ~0); 0.5 splits them.
    let threshold = 0.5;

    let mut input = ffi_input_stream(make_mixed_input_batches(32, 32, 7));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe { rype_classify_arrow_log_ratio(num, denom, &mut input, threshold, &mut out) };
    assert_eq!(rc, 0, "log_ratio failed: {}", last_error());
    let (_, baseline) = drain_log_ratios(out)?;
    assert_eq!(baseline.len(), 32, "one result per read");

    // Pin the premise: both branches must be represented, or the assertions
    // below would hold for a classifier that never took the fast path.
    let fast = baseline.iter().filter(|r| r.2 == FAST_NUM_HIGH).count();
    let exact = baseline.iter().filter(|r| r.2 == FAST_NONE).count();
    assert!(
        fast > 0 && exact > 0,
        "fixture must produce both fast-path and exact reads, got fast={} exact={}",
        fast,
        exact
    );
    // Fast-path reads are assigned +inf without a denominator pass.
    for row in baseline.iter().filter(|r| r.2 == FAST_NUM_HIGH) {
        assert!(
            row.1.is_infinite() && row.1 > 0.0,
            "fast-path read {} should be +inf, got {}",
            row.0,
            row.1
        );
    }

    // Same reads over 8 input batches, one group: identical ratios and identical
    // per-read fast-path flags.
    let mut input = ffi_input_stream(make_mixed_input_batches(32, 4, 7));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc = unsafe {
        rype_classify_arrow_log_ratio_ex(num, denom, &mut input, threshold, 32, &mut out)
    };
    assert_eq!(rc, 0, "log_ratio_ex failed: {}", last_error());
    let (one_group_batches, one_group) = drain_log_ratios(out)?;
    assert!(
        log_ratios_match(&one_group, &baseline),
        "batching changed the fast-path split"
    );
    assert_eq!(one_group_batches, 1);

    // Four groups of 8: the partition is per group, so this is the case where a
    // group-size dependency would surface.
    let mut input = ffi_input_stream(make_mixed_input_batches(32, 4, 7));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_log_ratio_ex(num, denom, &mut input, threshold, 8, &mut out) };
    assert_eq!(rc, 0, "log_ratio_ex failed: {}", last_error());
    let (multi_batches, multi) = drain_log_ratios(out)?;
    assert!(
        log_ratios_match(&multi, &baseline),
        "regrouping changed the fast-path split"
    );
    assert_eq!(multi_batches, 4, "expected 32/8 = 4 classification passes");

    // classify_batch_rows=0 must reproduce one pass per input batch, which no
    // log-ratio test covered before.
    let mut input = ffi_input_stream(make_mixed_input_batches(32, 4, 7));
    let mut out = FFI_ArrowArrayStream::empty();
    let rc =
        unsafe { rype_classify_arrow_log_ratio_ex(num, denom, &mut input, threshold, 0, &mut out) };
    assert_eq!(rc, 0, "log_ratio_ex failed: {}", last_error());
    let (per_batch_batches, per_batch) = drain_log_ratios(out)?;
    assert!(log_ratios_match(&per_batch, &baseline));
    assert_eq!(
        per_batch_batches, 8,
        "classify_batch_rows=0 must classify each input batch on its own"
    );

    rype_index_free(denom);
    rype_index_free(num);
    Ok(())
}
