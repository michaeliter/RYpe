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
    rype_classify_arrow, rype_classify_arrow_ex, rype_classify_arrow_log_ratio,
    rype_classify_arrow_log_ratio_ex, rype_get_last_error, rype_index_free, rype_index_load,
    rype_negative_set_create, rype_negative_set_free, RypeIndex, RypeNegativeSet,
};
use rype::{
    create_parquet_inverted_index, extract_into, BucketData, MinimizerWorkspace,
    ParquetWriteOptions,
};

const K: usize = 16;
const W: usize = 5;
const SALT: u64 = 0x12345;

/// Deterministic pseudo-DNA. Distinct `seed`s give distinct minimizer sets.
fn generate_sequence(len: usize, seed: u64) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bases[((state >> 33) % 4) as usize]
        })
        .collect()
}

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

fn ffi_input_stream(batches: Vec<RecordBatch>) -> FFI_ArrowArrayStream {
    let schema = input_schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    FFI_ArrowArrayStream::new(Box::new(reader))
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

/// Drain an output stream into (batch count, rows sorted for comparison).
fn drain_hits(out: FFI_ArrowArrayStream) -> Result<(usize, Vec<(i64, u32, f64)>)> {
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
fn drain_log_ratios(out: FFI_ArrowArrayStream) -> Result<(usize, Vec<(i64, f64, i32)>)> {
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
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((batch_count, rows))
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

    assert_eq!(
        accumulated, baseline,
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
    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut out) }?;
    assert_eq!(reader.schema(), rype::arrow::result_schema());

    rype_index_free(index);
    Ok(())
}
