//! Integration tests for Apache Arrow support.
//!
//! These tests verify end-to-end functionality of the Arrow integration,
//! including consistency with the regular API and streaming behavior.

#![cfg(feature = "arrow-ffi")]

use anyhow::Result;
use arrow::array::{Array, BinaryArray, Float64Array, Int64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::tempdir;

use rype::arrow::{
    batch_to_records, classify_arrow_batch_sharded, classify_arrow_batch_sharded_best_hit,
    hits_to_record_batch, result_schema, validate_input_schema, ShardedStreamClassifier, COL_ID,
    COL_PAIR_SEQUENCE, COL_SEQUENCE,
};
use rype::{
    classify_batch_sharded_merge_join, create_parquet_inverted_index, extract_into, BucketData,
    MinimizerWorkspace, ParquetWriteOptions, ShardedInvertedIndex,
};

mod common;
use common::generate_sequence;

/// Create a test batch with the expected schema.
fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]));

    let id_array = Int64Array::from(ids.to_vec());
    let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
}

/// Create a test batch with paired-end sequences.
fn make_test_batch_paired(ids: &[i64], seqs: &[&[u8]], pairs: &[Option<&[u8]>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
        Field::new(COL_PAIR_SEQUENCE, DataType::Binary, true),
    ]));

    let id_array = Int64Array::from(ids.to_vec());
    let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());
    let pair_array = BinaryArray::from_iter(pairs.iter().copied());

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(seq_array),
            Arc::new(pair_array),
        ],
    )
    .unwrap()
}

/// Create a test Parquet index with multiple buckets.
fn create_test_parquet_index() -> (tempfile::TempDir, ShardedInvertedIndex) {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("test.ryxdi");

    let mut ws = MinimizerWorkspace::new();

    // Bucket 1: Pattern starting with seed 0
    let ref_seq1 = generate_sequence(100, 0);
    extract_into(&ref_seq1, 16, 5, 0x12345, &mut ws);
    let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
    mins1.sort();
    mins1.dedup();

    // Bucket 2: Pattern starting with seed 2
    let ref_seq2 = generate_sequence(100, 2);
    extract_into(&ref_seq2, 16, 5, 0x12345, &mut ws);
    let mut mins2: Vec<u64> = ws.buffer.drain(..).collect();
    mins2.sort();
    mins2.dedup();

    let buckets = vec![
        BucketData {
            bucket_id: 1,
            bucket_name: "bucket1".to_string(),
            sources: vec!["ref1".to_string()],
            minimizers: mins1,
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "bucket2".to_string(),
            sources: vec!["ref2".to_string()],
            minimizers: mins2,
        },
    ];

    let options = ParquetWriteOptions::default();
    create_parquet_inverted_index(
        &index_path,
        buckets,
        16,
        5,
        0x12345,
        None,
        Some(&options),
        None,
    )
    .unwrap();

    let index = ShardedInvertedIndex::open(&index_path).unwrap();
    (dir, index)
}

#[test]
fn test_arrow_roundtrip_classification() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    // Query that matches bucket 1
    let query_seq = generate_sequence(100, 0);
    let batch = make_test_batch(&[101], &[&query_seq]);

    let result = classify_arrow_batch_sharded(&index, None, &batch, 0.0)?;

    assert!(result.num_rows() > 0, "Should have classification results");

    // Verify schema
    assert_eq!(result.schema(), result_schema());

    // Verify we can read the data
    let query_ids = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(query_ids.value(0), 101);

    Ok(())
}

#[test]
fn test_arrow_vs_regular_api_consistency() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();
    let threshold = 0.1;

    // Create test queries
    let query1 = generate_sequence(100, 0); // Matches bucket 1
    let query2 = generate_sequence(100, 2); // Matches bucket 2
    let query3 = generate_sequence(100, 1); // Different pattern

    // Regular API
    let records: Vec<rype::QueryRecord> = vec![
        (1, query1.as_slice(), None),
        (2, query2.as_slice(), None),
        (3, query3.as_slice(), None),
    ];
    let regular_hits = classify_batch_sharded_merge_join(&index, None, &records, threshold, None)?;

    // Arrow API
    let batch = make_test_batch(&[1, 2, 3], &[&query1, &query2, &query3]);
    let arrow_result = classify_arrow_batch_sharded(&index, None, &batch, threshold)?;

    // Results should be consistent
    assert_eq!(
        regular_hits.len(),
        arrow_result.num_rows(),
        "Arrow and regular API should produce same number of hits"
    );

    // Verify each regular hit is present in arrow results
    let arrow_query_ids = arrow_result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let arrow_bucket_ids = arrow_result
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let arrow_scores = arrow_result
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    for regular_hit in &regular_hits {
        let found = (0..arrow_result.num_rows()).any(|i| {
            arrow_query_ids.value(i) == regular_hit.query_id
                && arrow_bucket_ids.value(i) == regular_hit.bucket_id
                && (arrow_scores.value(i) - regular_hit.score).abs() < 1e-10
        });
        assert!(
            found,
            "Regular hit {:?} not found in Arrow results",
            regular_hit
        );
    }

    Ok(())
}

#[test]
fn test_arrow_with_paired_end() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    let seq1 = generate_sequence(80, 0);
    let pair1 = generate_sequence(80, 0);
    let seq2 = generate_sequence(80, 2);

    let batch = make_test_batch_paired(&[1, 2], &[&seq1, &seq2], &[Some(pair1.as_slice()), None]);

    let result = classify_arrow_batch_sharded(&index, None, &batch, 0.0)?;

    // Should have results for both queries
    assert!(result.num_rows() >= 2);

    Ok(())
}

#[test]
fn test_arrow_large_batch() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    // Create a large batch
    let num_queries = 1000;
    let ids: Vec<i64> = (0..num_queries).collect();
    let sequences: Vec<Vec<u8>> = (0..num_queries)
        .map(|i| generate_sequence(100, (i % 4) as u64))
        .collect();
    let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();

    let batch = make_test_batch(&ids, &seq_refs);

    let result = classify_arrow_batch_sharded(&index, None, &batch, 0.0)?;

    // Should handle large batches without error
    assert!(result.num_rows() > 0);

    Ok(())
}

#[test]
fn test_arrow_streaming_multiple_batches() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();
    let classifier = ShardedStreamClassifier::new(&index, None, 0.0);

    // Create multiple batches
    let batch1 = make_test_batch(
        &[1, 2],
        &[&generate_sequence(100, 0), &generate_sequence(100, 1)],
    );
    let batch2 = make_test_batch(
        &[3, 4],
        &[&generate_sequence(100, 2), &generate_sequence(100, 3)],
    );
    let batch3 = make_test_batch(&[5], &[&generate_sequence(100, 0)]);

    let input_batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> =
        vec![Ok(batch1), Ok(batch2), Ok(batch3)];

    let results: Vec<_> = classifier
        .classify_iter(input_batches.into_iter())
        .collect();

    assert_eq!(results.len(), 3, "Should have one result per input batch");
    for result in results {
        assert!(result.is_ok(), "Each batch should classify successfully");
    }

    Ok(())
}

#[test]
fn test_arrow_threshold_filtering() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    // A query that fully matches bucket 1 scores 1.0 and a query that matches
    // nothing is never reported, so neither can distinguish one threshold from
    // another. Use a chimera — half of bucket 1's reference, half novel — which
    // lands strictly between 0 and 1 and so is reported at a low threshold and
    // filtered at a high one.
    let ref_seq = generate_sequence(100, 0);
    let novel = generate_sequence(100, 9_001);
    let mut chimera = ref_seq[..50].to_vec();
    chimera.extend_from_slice(&novel[..50]);
    let batch = make_test_batch(&[1], &[&chimera]);

    let low_result = classify_arrow_batch_sharded(&index, None, &batch, 0.0)?;
    let high_result = classify_arrow_batch_sharded(&index, None, &batch, 0.99)?;

    // Pin the premise: the chimera must actually score in between, otherwise the
    // comparison below would hold for uninteresting reasons.
    let scores = low_result
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(low_result.num_rows() > 0, "chimera should hit bucket 1");
    let score = scores.value(0);
    assert!(
        score > 0.0 && score < 0.99,
        "chimera score {} must be strictly between the two thresholds for this \
         test to discriminate",
        score
    );

    // Strictly fewer, not merely "no more": a classifier that ignored the
    // threshold would satisfy >= but not >.
    assert!(
        low_result.num_rows() > high_result.num_rows(),
        "raising the threshold above the score must drop the hit (low={}, high={})",
        low_result.num_rows(),
        high_result.num_rows()
    );

    Ok(())
}

#[test]
fn test_arrow_empty_batch() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]));
    let empty_batch = RecordBatch::new_empty(schema);

    let result = classify_arrow_batch_sharded(&index, None, &empty_batch, 0.1)?;

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.schema(), result_schema());

    Ok(())
}

#[test]
fn test_arrow_schema_validation() -> Result<()> {
    // Valid schema
    let valid_schema = Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]);
    assert!(validate_input_schema(&valid_schema).is_ok());

    // Invalid: wrong ID type
    let invalid_schema = Schema::new(vec![
        Field::new(COL_ID, DataType::Utf8, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]);
    assert!(validate_input_schema(&invalid_schema).is_err());

    // Invalid: missing sequence
    let missing_seq = Schema::new(vec![Field::new(COL_ID, DataType::Int64, false)]);
    assert!(validate_input_schema(&missing_seq).is_err());

    Ok(())
}

#[test]
fn test_arrow_zero_copy_verification() -> Result<()> {
    let seq_data = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let batch = make_test_batch(&[1], &[seq_data]);

    let records = batch_to_records(&batch)?;

    // Get pointer to sequence in record
    let record_ptr = records[0].1.as_ptr();

    // Get pointer to sequence in Arrow array
    let seq_col = batch.column(1);
    let binary_arr = seq_col.as_any().downcast_ref::<BinaryArray>().unwrap();
    let arrow_ptr = binary_arr.value(0).as_ptr();

    // Verify zero-copy: pointers should be identical
    assert_eq!(
        record_ptr, arrow_ptr,
        "batch_to_records should provide zero-copy access to Arrow buffers"
    );

    Ok(())
}

#[test]
fn test_arrow_hits_to_batch_roundtrip() -> Result<()> {
    let hits = vec![
        rype::HitResult {
            query_id: 1,
            bucket_id: 10,
            score: 0.95,
        },
        rype::HitResult {
            query_id: 2,
            bucket_id: 20,
            score: 0.85,
        },
        rype::HitResult {
            query_id: 3,
            bucket_id: 10,
            score: 0.75,
        },
    ];

    let batch = hits_to_record_batch(hits.clone())?;

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);

    // Verify data
    let query_ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let bucket_ids = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let scores = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    for (i, hit) in hits.iter().enumerate() {
        assert_eq!(query_ids.value(i), hit.query_id);
        assert_eq!(bucket_ids.value(i), hit.bucket_id);
        assert!((scores.value(i) - hit.score).abs() < 1e-10);
    }

    Ok(())
}

/// `classify_arrow_batch_sharded_best_hit` is the library counterpart of the C
/// API's best-hit entry points and shares `classify_arrow_from_extracted` with
/// them, so this is also the non-FFI check on that shared reduction.
#[test]
fn test_arrow_best_hit_keeps_one_row_per_query() -> Result<()> {
    let (_dir, index) = create_test_parquet_index();

    // Chimeras of the two references so each query hits both buckets with
    // different scores — otherwise there is nothing for best-hit to choose
    // between and the reduction would be untestable.
    let ref1 = generate_sequence(100, 0);
    let ref2 = generate_sequence(100, 2);
    let mut q1 = ref1[..70].to_vec();
    q1.extend_from_slice(&ref2[..30]);
    let mut q2 = ref2[..70].to_vec();
    q2.extend_from_slice(&ref1[..30]);
    let batch = make_test_batch(&[1, 2], &[&q1, &q2]);

    let all = classify_arrow_batch_sharded(&index, None, &batch, 0.0)?;
    let best = classify_arrow_batch_sharded_best_hit(&index, None, &batch, 0.0)?;

    assert_eq!(best.schema(), result_schema());
    assert!(
        all.num_rows() > best.num_rows(),
        "each query should hit both buckets ({} rows) and reduce to one ({} rows)",
        all.num_rows(),
        best.num_rows()
    );

    let ids = best
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mut seen: Vec<i64> = (0..best.num_rows()).map(|i| ids.value(i)).collect();
    let total = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), total, "best-hit emitted a query id twice");

    // The kept row must be the query's maximum score, not an arbitrary one.
    let best_scores = best
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let all_ids = all.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let all_scores = all
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..best.num_rows() {
        let qid = ids.value(i);
        let max = (0..all.num_rows())
            .filter(|&j| all_ids.value(j) == qid)
            .map(|j| all_scores.value(j))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (best_scores.value(i) - max).abs() < 1e-12,
            "query {} kept score {} but its best is {}",
            qid,
            best_scores.value(i),
            max
        );
    }

    Ok(())
}
