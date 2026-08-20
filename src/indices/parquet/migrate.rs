//! v1 -> v2 (COO -> CSR) shard format migration.
//!
//! v1 shards are already globally sorted by minimizer, so migrating one is a
//! single streaming pass with no re-sort and no re-extraction from source
//! genomes: `streaming::migrate_shard_v1_to_v2` reads pairs in order and
//! regroups them into CSR rows. Shard boundaries, counts, and min/max
//! minimizer values are unchanged by the conversion, so this only touches
//! physical layout, never which minimizer maps to which shard or bucket.
//!
//! See the "ryxdi v2" plan (`~/.claude/plans/we-previously-suggested-improving-async-zebra.md`)
//! and `scratch/PHASE0-RESULTS.md` for why this format exists.

use std::fs;
use std::path::Path;

use crate::error::{Result, RypeError};

use super::files;
use super::manifest::{create_index_directory, InvertedManifest, ParquetManifest};
use super::options::ParquetWriteOptions;
use super::streaming::migrate_shard_v1_to_v2;
use super::ParquetShardFormat;

/// Migrate a v1 (COO) `.ryxdi` index at `input_dir` to v2 (CSR) at
/// `output_dir`. `output_dir` must not already exist as an index (this
/// creates a fresh directory; it does not merge into or overwrite one).
///
/// Returns an error if `input_dir` is already v2 — a script chaining
/// `migrate` should be able to tell "already migrated" from "it worked" by
/// the exit code, not by diffing output.
pub fn migrate_v1_to_v2(
    input_dir: &Path,
    output_dir: &Path,
    options: &ParquetWriteOptions,
) -> Result<ParquetManifest> {
    let manifest = ParquetManifest::load(input_dir)?;
    let inverted = manifest.inverted.as_ref().ok_or_else(|| {
        RypeError::validation(format!(
            "{}: manifest has no [inverted] section, nothing to migrate",
            input_dir.display()
        ))
    })?;

    if inverted.format == ParquetShardFormat::Csr {
        return Err(RypeError::validation(format!(
            "{} is already v2 (CSR) format; nothing to migrate",
            input_dir.display()
        )));
    }

    create_index_directory(output_dir)?;

    // buckets.parquet is unaffected by the shard row-layout format (it only
    // describes buckets, never minimizers) — copy as-is rather than
    // round-tripping through read/write.
    let buckets_src = input_dir.join(files::BUCKETS);
    let buckets_dst = output_dir.join(files::BUCKETS);
    fs::copy(&buckets_src, &buckets_dst)
        .map_err(|e| RypeError::io(buckets_dst, "copy buckets.parquet", e))?;

    for shard in &inverted.shards {
        if shard.num_entries == 0 {
            // Empty shards are never written to disk in the first place (see
            // stream_to_parquet_shards's empty-index case) — nothing to read.
            continue;
        }
        let input_shard = input_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(shard.shard_id));
        let output_shard = output_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(shard.shard_id));
        migrate_shard_v1_to_v2(&input_shard, &output_shard, options)?;
    }

    // Shard boundaries, counts, and min/max are unchanged by the format
    // conversion (the set of (minimizer, bucket_id) pairs a shard holds is
    // identical, only their on-disk row layout differs) — copy them as-is
    // rather than recomputing.
    let new_manifest = ParquetManifest {
        magic: manifest.magic.clone(),
        format_version: manifest.format_version,
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
        source_hash: manifest.source_hash,
        num_buckets: manifest.num_buckets,
        total_minimizers: manifest.total_minimizers,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Csr,
            num_shards: inverted.num_shards,
            total_entries: inverted.total_entries,
            has_overlapping_shards: inverted.has_overlapping_shards,
            shards: inverted.shards.clone(),
        }),
    };
    new_manifest.save(output_dir)?;

    Ok(new_manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::{create_parquet_inverted_index, BucketData};
    use crate::indices::sharded::ShardedInvertedIndex;
    use tempfile::TempDir;

    #[test]
    fn test_migrate_v1_shard_round_trip() {
        use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::collections::HashMap;
        use std::sync::Arc;

        let tmp = TempDir::new().unwrap();
        let input_dir = tmp.path().join("v1.ryxdi");
        let output_dir = tmp.path().join("v2.ryxdi");
        create_index_directory(&input_dir).unwrap();

        // Hand-write a genuine v1 (COO) shard, bypassing ShardWriter (which
        // is CSR-only now) — this is what a pre-v2 binary's output looked
        // like on disk.
        let pairs: Vec<(u64, u32)> = vec![(100, 0), (150, 1), (200, 0), (200, 1), (300, 0)];
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));
        let shard_path = input_dir.join(files::INVERTED_DIR).join("shard.0.parquet");
        let file = std::fs::File::create(&shard_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let min_array: ArrayRef =
            Arc::new(UInt64Array::from(pairs.iter().map(|p| p.0).collect::<Vec<_>>()));
        let bid_array: ArrayRef =
            Arc::new(UInt32Array::from(pairs.iter().map(|p| p.1).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema, vec![min_array, bid_array]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(0u32, "a".to_string());
        bucket_names.insert(1u32, "b".to_string());
        let mut bucket_sources = HashMap::new();
        bucket_sources.insert(0u32, vec!["a.fna".to_string()]);
        bucket_sources.insert(1u32, vec!["b.fna".to_string()]);
        crate::indices::parquet::write_buckets_parquet(
            &input_dir,
            &bucket_names,
            &bucket_sources,
            None,
        )
        .unwrap();

        let manifest = ParquetManifest {
            magic: super::super::FORMAT_MAGIC.to_string(),
            format_version: super::super::FORMAT_VERSION,
            k: 64,
            w: 50,
            salt: 12345,
            source_hash: 0xDEAD,
            num_buckets: 2,
            total_minimizers: pairs.len() as u64,
            inverted: Some(InvertedManifest {
                format: ParquetShardFormat::Parquet,
                num_shards: 1,
                total_entries: pairs.len() as u64,
                has_overlapping_shards: false,
                shards: vec![crate::indices::parquet::InvertedShardInfo {
                    shard_id: 0,
                    min_minimizer: 100,
                    max_minimizer: 300,
                    num_entries: pairs.len() as u64,
                }],
            }),
        };
        manifest.save(&input_dir).unwrap();

        let migrated = migrate_v1_to_v2(&input_dir, &output_dir, &ParquetWriteOptions::default())
            .expect("migrate should succeed on a genuine v1 shard");
        let migrated_inverted = migrated.inverted.as_ref().unwrap();
        assert_eq!(migrated_inverted.format, ParquetShardFormat::Csr);
        // Counts, ranges, and bucket metadata are copied through unchanged.
        assert_eq!(migrated_inverted.total_entries, pairs.len() as u64);
        assert_eq!(migrated_inverted.shards[0].min_minimizer, 100);
        assert_eq!(migrated_inverted.shards[0].max_minimizer, 300);

        // The migrated index classifies identically to the original: every
        // (minimizer, bucket_id) pair is preserved.
        let opened = ShardedInvertedIndex::open(&output_dir).unwrap();
        let all_minimizers: Vec<u64> = pairs.iter().map(|p| p.0).collect();
        let loaded = opened
            .load_shard_for_query(0, &all_minimizers, None)
            .unwrap();
        let mut loaded_pairs: Vec<(u64, u32)> = Vec::new();
        for (i, &m) in loaded.minimizers().iter().enumerate() {
            for &b in &loaded.bucket_ids()[loaded.offsets()[i] as usize..loaded.offsets()[i + 1] as usize] {
                loaded_pairs.push((m, b));
            }
        }
        let mut expected = pairs.clone();
        expected.sort();
        loaded_pairs.sort();
        assert_eq!(loaded_pairs, expected);
    }

    #[test]
    fn test_migrate_rejects_already_v2() {
        let tmp = TempDir::new().unwrap();
        let input_dir = tmp.path().join("v2.ryxdi");
        let output_dir = tmp.path().join("v2_out.ryxdi");

        let buckets = vec![BucketData {
            bucket_id: 0,
            bucket_name: "a".to_string(),
            sources: vec![],
            minimizers: vec![1, 2, 3],
        }];
        create_parquet_inverted_index(&input_dir, buckets, 64, 50, 1, None, None, None).unwrap();

        let result = migrate_v1_to_v2(&input_dir, &output_dir, &ParquetWriteOptions::default());
        assert!(result.is_err(), "migrating an already-v2 index must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("already v2"),
            "error should explain why: {msg}"
        );
    }
}
