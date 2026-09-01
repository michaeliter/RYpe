//! Incremental bucket updates: add new sequences to one existing bucket of a
//! `.ryxdi` index without rebuilding the other buckets.
//!
//! See the design notes in the project plan for the full picture. Phase A
//! (extracting the delta from new FASTX files) lives in
//! `commands::build_bucket_streaming_isolated`; this module implements
//! Phase B (shard selection + merge) and Phase C (assembling the output
//! index), driven by [`apply_bucket_addition`].

use crate::error::{Result, RypeError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::manifest::{InvertedManifest, InvertedShardInfo, ParquetManifest};
use super::merge::read_shard_pairs;
use super::options::ParquetWriteOptions;
use super::streaming::{merge_shard_paths_into, ShardAccumulator};
use super::{files, read_buckets_parquet, write_buckets_parquet, FORMAT_MAGIC, FORMAT_VERSION};

/// Read the `[min, max]` range of the `bucket_id` column of a shard file from
/// its Parquet footer, without loading any row data.
///
/// Mirrors `InvertedIndex::get_parquet_row_group_stats`
/// (`indices/inverted/shard_parquet.rs`), which does the same for the
/// `minimizer` column, but aggregates across all row groups into a single
/// file-level range since callers only need "does this shard touch bucket
/// X," not a per-row-group breakdown.
///
/// Returns `(0, u32::MAX)` — i.e. "assume relevant" — whenever statistics are
/// absent for any row group. This is the safe fallback: a shard incorrectly
/// treated as relevant is merged (extra work, still correct); a shard
/// incorrectly treated as irrelevant would silently drop rows.
pub(crate) fn bucket_id_range(path: &Path) -> Result<(u32, u32)> {
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader;
    use parquet::file::statistics::Statistics;
    use std::fs::File;

    const FULL_RANGE: (u32, u32) = (0, u32::MAX);
    const BUCKET_ID_COLUMN: usize = 1;

    let file = File::open(path).map_err(|e| RypeError::io(path.to_path_buf(), "open shard", e))?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    if metadata.num_row_groups() == 0 {
        return Ok(FULL_RANGE);
    }

    let mut overall_min = u32::MAX;
    let mut overall_max = 0u32;

    for rg_idx in 0..metadata.num_row_groups() {
        let col_meta = metadata.row_group(rg_idx).column(BUCKET_ID_COLUMN);
        let Some(Statistics::Int32(stats)) = col_meta.statistics() else {
            return Ok(FULL_RANGE);
        };
        let (Some(min), Some(max)) = (stats.min_opt(), stats.max_opt()) else {
            return Ok(FULL_RANGE);
        };
        overall_min = overall_min.min(*min as u32);
        overall_max = overall_max.max(*max as u32);
    }

    Ok((overall_min, overall_max))
}

/// Result of a successful [`apply_bucket_addition`] call.
#[derive(Debug, Clone)]
pub struct BucketUpdateStats {
    /// Minimizers extracted from the new files that were not already in the
    /// target bucket (i.e. actually added).
    pub novel_minimizers: u64,
    /// Minimizers extracted from the new files that the target bucket
    /// already had (deduped away, not double-counted).
    pub already_present: u64,
    /// Number of the target bucket's original shards that had to be
    /// re-read and re-written (any shard whose `bucket_id` range includes
    /// the target bucket).
    pub shards_rewritten: usize,
    /// Number of shards carried over untouched via hard link/copy.
    pub shards_carried_over: usize,
    /// Total (minimizer, bucket_id) entries in the output index.
    pub total_minimizers: u64,
}

/// Exact per-bucket entry counts for one shard file.
///
/// O(1) for a bucket-exclusive shard (`range.0 == range.1`, the common case
/// for `index from-config`-built multi-bucket indices — see the module docs
/// for why). Falls back to a full read only for a shard spanning multiple
/// buckets, which only occurs for non-bucket-exclusive layouts (e.g.
/// `index create`'s range-partitioned shards).
fn shard_bucket_counts(
    path: &Path,
    range: (u32, u32),
    num_entries: u64,
) -> Result<HashMap<u32, u64>> {
    if range.0 == range.1 {
        let mut counts = HashMap::with_capacity(1);
        counts.insert(range.0, num_entries);
        return Ok(counts);
    }
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for (_, bucket_id) in read_shard_pairs(path)? {
        *counts.entry(bucket_id).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Hard-link `src` to `dst`, falling back to a byte copy when the two paths
/// are on different filesystems (`EXDEV`) — e.g. the index directory and the
/// output directory live on different mounts.
///
/// Safe to use for shard files specifically because nothing in this codebase
/// ever mutates a shard file in place: shard files are only ever created
/// whole (by a `ShardAccumulator` flush) and deleted whole, so a hard-linked
/// copy can never be corrupted by a write through the other name.
fn link_or_copy_shard(src: &Path, dst: &Path) -> Result<()> {
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.raw_os_error() == Some(libc::EXDEV) => std::fs::copy(src, dst)
            .map(|_| ())
            .map_err(|e| RypeError::io(dst.to_path_buf(), "copy shard across filesystems", e)),
        Err(e) => Err(RypeError::io(dst.to_path_buf(), "hard-link shard", e)),
    }
}

/// Result of classifying an index's existing shards against a target bucket
/// (Phase B), shared by both the `-o` and `--in-place` assembly paths.
struct ShardClassification {
    /// Shards whose `bucket_id` range includes the target bucket; must be
    /// re-merged with the delta.
    relevant: Vec<InvertedShardInfo>,
    /// Shards that don't touch the target bucket; carried over verbatim.
    untouched: Vec<InvertedShardInfo>,
    /// Exact count of the target bucket's entries before the update, read
    /// only from the (small) relevant shards.
    existing_target_count: u64,
    /// Full per-bucket counts for every untouched shard, needed to recompute
    /// `source_hash` over the whole output index.
    untouched_bucket_counts: HashMap<u32, u64>,
}

/// Phase B: classify `shards` (all living in `inverted_dir`) as relevant to
/// `target_bucket_id` or untouched, and compute the counts
/// [`apply_bucket_addition`]/the in-place path need to report novel-minimizer
/// stats and recompute `source_hash`.
fn classify_shards(
    inverted_dir: &Path,
    shards: &[InvertedShardInfo],
    target_bucket_id: u32,
) -> Result<ShardClassification> {
    let mut untouched: Vec<InvertedShardInfo> = Vec::new();
    let mut relevant: Vec<InvertedShardInfo> = Vec::new();
    for info in shards {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        let range = bucket_id_range(&path)?;
        if range.0 <= target_bucket_id && target_bucket_id <= range.1 {
            relevant.push(*info);
        } else {
            untouched.push(*info);
        }
    }

    let mut existing_target_count: u64 = 0;
    for info in &relevant {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        let range = bucket_id_range(&path)?;
        let counts = shard_bucket_counts(&path, range, info.num_entries)?;
        existing_target_count += counts.get(&target_bucket_id).copied().unwrap_or(0);
    }

    let mut untouched_bucket_counts: HashMap<u32, u64> = HashMap::new();
    for info in &untouched {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        let range = bucket_id_range(&path)?;
        let counts = shard_bucket_counts(&path, range, info.num_entries)?;
        for (bucket_id, count) in counts {
            *untouched_bucket_counts.entry(bucket_id).or_insert(0) += count;
        }
    }

    Ok(ShardClassification {
        relevant,
        untouched,
        existing_target_count,
        untouched_bucket_counts,
    })
}

/// Add new minimizers to one existing bucket of a `.ryxdi` index, writing the
/// result to `output_path`, without touching any other bucket's shards.
///
/// `delta_dir` must be a Parquet index directory (as produced by
/// `commands::build_bucket_streaming_isolated` followed by
/// `consolidate_streaming_shards`) whose shards are already sorted, deduped,
/// and tagged with `target_bucket_id`; `delta_shards` describes those shard
/// files. `new_sources` are the source labels to append to the target
/// bucket's `sources` list.
///
/// See the module docs and the project plan for the three-phase design
/// (extract delta → select + merge relevant shards → assemble output).
#[allow(clippy::too_many_arguments)]
pub fn apply_bucket_addition(
    index_path: &Path,
    target_bucket_id: u32,
    delta_dir: &Path,
    delta_shards: &[InvertedShardInfo],
    new_sources: &[String],
    output_path: &Path,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<BucketUpdateStats> {
    let manifest = ParquetManifest::load(index_path)?;
    let (bucket_names, mut bucket_sources, bucket_file_stats) = read_buckets_parquet(index_path)?;

    if !bucket_names.contains_key(&target_bucket_id) {
        return Err(RypeError::validation(format!(
            "bucket {} not found in index {}",
            target_bucket_id,
            index_path.display()
        )));
    }

    let inverted = manifest.inverted.as_ref().ok_or_else(|| {
        RypeError::validation(format!(
            "index {} has no inverted section",
            index_path.display()
        ))
    })?;

    let source_inverted_dir = index_path.join(files::INVERTED_DIR);
    let delta_inverted_dir = delta_dir.join(files::INVERTED_DIR);

    let classification = classify_shards(&source_inverted_dir, &inverted.shards, target_bucket_id)?;
    let ShardClassification {
        relevant,
        untouched,
        existing_target_count,
        untouched_bucket_counts: mut bucket_minimizer_counts,
    } = classification;

    let delta_total: u64 = delta_shards.iter().map(|s| s.num_entries).sum();

    // Phase C: assemble the output index. Untouched shards get ids
    // 0..n_untouched (hard-linked as-is); the merge output starts right
    // after, so shard ids come out contiguous with no separate rename pass.
    super::create_index_directory(output_path)?;
    let output_inverted_dir = output_path.join(files::INVERTED_DIR);

    let n_untouched = untouched.len() as u32;
    let mut output_shard_infos: Vec<InvertedShardInfo> = Vec::with_capacity(untouched.len());
    for (i, info) in untouched.iter().enumerate() {
        let src = source_inverted_dir.join(files::inverted_shard(info.shard_id));
        let new_id = i as u32;
        let dst = output_inverted_dir.join(files::inverted_shard(new_id));
        link_or_copy_shard(&src, &dst)?;
        output_shard_infos.push(InvertedShardInfo {
            shard_id: new_id,
            min_minimizer: info.min_minimizer,
            max_minimizer: info.max_minimizer,
            num_entries: info.num_entries,
        });
    }

    let merge_paths: Vec<PathBuf> = relevant
        .iter()
        .map(|info| source_inverted_dir.join(files::inverted_shard(info.shard_id)))
        .chain(
            delta_shards
                .iter()
                .map(|info| delta_inverted_dir.join(files::inverted_shard(info.shard_id))),
        )
        .collect();

    let mut accumulator = ShardAccumulator::with_start_shard_id(
        output_path,
        max_shard_bytes,
        n_untouched,
        Some(options),
    );
    let merge_counts = merge_shard_paths_into(&merge_paths, &mut accumulator)?;
    let merged_infos = accumulator.finish()?;
    output_shard_infos.extend(merged_infos);

    for (bucket_id, count) in &merge_counts.bucket_counts {
        bucket_minimizer_counts.insert(*bucket_id, *count);
    }

    let total_entries: u64 = output_shard_infos.iter().map(|s| s.num_entries).sum();
    let has_overlapping_shards = inverted.has_overlapping_shards || !untouched.is_empty();

    if !new_sources.is_empty() {
        bucket_sources
            .entry(target_bucket_id)
            .or_default()
            .extend(new_sources.iter().cloned());
    }
    if bucket_file_stats
        .as_ref()
        .is_some_and(|stats| stats.contains_key(&target_bucket_id))
    {
        log::warn!(
            "bucket-update: file-length statistics for bucket {} are now stale \
             (median/stdev cannot be recomputed from the new files alone)",
            target_bucket_id
        );
    }

    write_buckets_parquet(
        output_path,
        &bucket_names,
        &bucket_sources,
        bucket_file_stats.as_ref(),
    )?;

    let source_hash_counts: HashMap<u32, usize> = bucket_minimizer_counts
        .iter()
        .map(|(&id, &count)| (id, count as usize))
        .collect();
    let source_hash = super::compute_source_hash(&source_hash_counts);

    let output_manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
        source_hash,
        num_buckets: bucket_names.len() as u32,
        total_minimizers: total_entries,
        inverted: Some(InvertedManifest {
            format: manifest
                .inverted
                .as_ref()
                .map(|m| m.format)
                .unwrap_or_default(),
            num_shards: output_shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards,
            shards: output_shard_infos,
        }),
    };
    output_manifest.save(output_path)?;

    let merged_target_count = merge_counts
        .bucket_counts
        .get(&target_bucket_id)
        .copied()
        .unwrap_or(0);
    let novel_minimizers = merged_target_count.saturating_sub(existing_target_count);
    let already_present = delta_total.saturating_sub(novel_minimizers);

    Ok(BucketUpdateStats {
        novel_minimizers,
        already_present,
        shards_rewritten: relevant.len(),
        shards_carried_over: untouched.len(),
        total_minimizers: total_entries,
    })
}

/// Atomically write `manifest` to `index_dir/manifest.toml`: serialize, write
/// to a `.tmp` sibling, then `rename` over the real path. The rename is the
/// single commit point — a crash before it leaves the original manifest (and
/// therefore the original index) completely intact.
fn save_manifest_atomically(manifest: &ParquetManifest, index_dir: &Path) -> Result<()> {
    let final_path = index_dir.join(files::MANIFEST);
    let tmp_path = index_dir.join(format!("{}.tmp", files::MANIFEST));

    let toml_str = toml::to_string_pretty(manifest)
        .map_err(|e| RypeError::encoding(format!("serialize manifest: {}", e)))?;
    std::fs::write(&tmp_path, &toml_str)
        .map_err(|e| RypeError::io(tmp_path.clone(), "write temp manifest", e))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| RypeError::io(final_path, "commit manifest", e))?;
    Ok(())
}

/// Like [`apply_bucket_addition`] but updates `index_path` itself instead of
/// writing to a new directory.
///
/// New merged shards are written directly into `index_path`'s own
/// `inverted/` directory, under shard ids starting at one past the highest
/// id already in use — so they can never collide with a shard file that
/// still exists. `buckets.parquet` is overwritten first (harmless if a crash
/// follows: a retried update re-derives the same sources from the
/// unchanged manifest and shard files), then `manifest.toml` is replaced via
/// a write-to-temp-then-rename helper — that rename is the actual commit point.
/// Only after it succeeds are the superseded "relevant" shard files deleted;
/// a crash at any point before the rename leaves a few harmless orphan shard
/// files next to a still-valid, unchanged index.
pub fn apply_bucket_addition_in_place(
    index_path: &Path,
    target_bucket_id: u32,
    delta_dir: &Path,
    delta_shards: &[InvertedShardInfo],
    new_sources: &[String],
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<BucketUpdateStats> {
    let manifest = ParquetManifest::load(index_path)?;
    let (bucket_names, mut bucket_sources, bucket_file_stats) = read_buckets_parquet(index_path)?;

    if !bucket_names.contains_key(&target_bucket_id) {
        return Err(RypeError::validation(format!(
            "bucket {} not found in index {}",
            target_bucket_id,
            index_path.display()
        )));
    }

    let inverted = manifest.inverted.as_ref().ok_or_else(|| {
        RypeError::validation(format!(
            "index {} has no inverted section",
            index_path.display()
        ))
    })?;

    let inverted_dir = index_path.join(files::INVERTED_DIR);
    let delta_inverted_dir = delta_dir.join(files::INVERTED_DIR);

    let classification = classify_shards(&inverted_dir, &inverted.shards, target_bucket_id)?;
    let ShardClassification {
        relevant,
        untouched,
        existing_target_count,
        untouched_bucket_counts: mut bucket_minimizer_counts,
    } = classification;

    let delta_total: u64 = delta_shards.iter().map(|s| s.num_entries).sum();

    let next_shard_id = inverted
        .shards
        .iter()
        .map(|s| s.shard_id)
        .max()
        .map_or(0, |m| m + 1);

    let merge_paths: Vec<PathBuf> = relevant
        .iter()
        .map(|info| inverted_dir.join(files::inverted_shard(info.shard_id)))
        .chain(
            delta_shards
                .iter()
                .map(|info| delta_inverted_dir.join(files::inverted_shard(info.shard_id))),
        )
        .collect();

    let mut accumulator = ShardAccumulator::with_start_shard_id(
        index_path,
        max_shard_bytes,
        next_shard_id,
        Some(options),
    );
    let merge_counts = merge_shard_paths_into(&merge_paths, &mut accumulator)?;
    let merged_infos = accumulator.finish()?;

    for (bucket_id, count) in &merge_counts.bucket_counts {
        bucket_minimizer_counts.insert(*bucket_id, *count);
    }

    let mut output_shard_infos = untouched.clone();
    output_shard_infos.extend(merged_infos);
    let total_entries: u64 = output_shard_infos.iter().map(|s| s.num_entries).sum();
    let has_overlapping_shards = inverted.has_overlapping_shards || !untouched.is_empty();

    if !new_sources.is_empty() {
        bucket_sources
            .entry(target_bucket_id)
            .or_default()
            .extend(new_sources.iter().cloned());
    }
    if bucket_file_stats
        .as_ref()
        .is_some_and(|stats| stats.contains_key(&target_bucket_id))
    {
        log::warn!(
            "bucket-update: file-length statistics for bucket {} are now stale \
             (median/stdev cannot be recomputed from the new files alone)",
            target_bucket_id
        );
    }

    write_buckets_parquet(
        index_path,
        &bucket_names,
        &bucket_sources,
        bucket_file_stats.as_ref(),
    )?;

    let source_hash_counts: HashMap<u32, usize> = bucket_minimizer_counts
        .iter()
        .map(|(&id, &count)| (id, count as usize))
        .collect();
    let source_hash = super::compute_source_hash(&source_hash_counts);

    let output_manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
        source_hash,
        num_buckets: bucket_names.len() as u32,
        total_minimizers: total_entries,
        inverted: Some(InvertedManifest {
            format: inverted.format,
            num_shards: output_shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards,
            shards: output_shard_infos,
        }),
    };
    save_manifest_atomically(&output_manifest, index_path)?;

    // Commit point has passed: the superseded shards are no longer
    // referenced by any manifest. Best-effort cleanup — an orphan left by a
    // failed removal is harmless disk usage, not a correctness issue.
    for info in &relevant {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        let _ = std::fs::remove_file(&path);
    }

    let merged_target_count = merge_counts
        .bucket_counts
        .get(&target_bucket_id)
        .copied()
        .unwrap_or(0);
    let novel_minimizers = merged_target_count.saturating_sub(existing_target_count);
    let already_present = delta_total.saturating_sub(novel_minimizers);

    Ok(BucketUpdateStats {
        novel_minimizers,
        already_present,
        shards_rewritten: relevant.len(),
        shards_carried_over: untouched.len(),
        total_minimizers: total_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::super::manifest::ParquetShardFormat;
    use super::super::streaming::write_shard_from_pairs;
    use super::*;
    use tempfile::TempDir;

    /// Build a `.ryxdi` directly from raw `(minimizer, bucket_id)` pairs, one
    /// shard per bucket (bucket-exclusive, matching what `index from-config`
    /// produces for a real multi-bucket build). Bypasses
    /// `create_parquet_inverted_index`'s size-based sharding so tests can
    /// exercise the "carry untouched shards over" path deterministically
    /// instead of getting whatever shard layout the size heuristic picks for
    /// tiny fixtures.
    fn build_fixture_index(
        dir: &Path,
        k: usize,
        w: usize,
        salt: u64,
        buckets: &[(u32, &str, Vec<u64>)],
    ) {
        std::fs::create_dir_all(dir).unwrap();
        super::super::create_index_directory(dir).unwrap();

        let mut bucket_names = HashMap::new();
        let mut bucket_sources = HashMap::new();
        let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
        let mut shard_infos = Vec::new();

        for (shard_id, (bucket_id, name, minimizers)) in buckets.iter().enumerate() {
            bucket_names.insert(*bucket_id, name.to_string());
            bucket_sources.insert(*bucket_id, vec!["orig_source".to_string()]);
            bucket_minimizer_counts.insert(*bucket_id, minimizers.len());

            let pairs: Vec<(u64, u32)> = minimizers.iter().map(|&m| (m, *bucket_id)).collect();
            let path = dir
                .join(files::INVERTED_DIR)
                .join(files::inverted_shard(shard_id as u32));
            write_shard_from_pairs(&path, &pairs, &ParquetWriteOptions::default()).unwrap();

            shard_infos.push(InvertedShardInfo {
                shard_id: shard_id as u32,
                min_minimizer: minimizers.iter().copied().min().unwrap_or(0),
                max_minimizer: minimizers.iter().copied().max().unwrap_or(0),
                num_entries: minimizers.len() as u64,
            });
        }

        write_buckets_parquet(dir, &bucket_names, &bucket_sources, None).unwrap();

        let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();
        let manifest = ParquetManifest {
            magic: FORMAT_MAGIC.to_string(),
            format_version: FORMAT_VERSION,
            k,
            w,
            salt,
            source_hash: super::super::compute_source_hash(&bucket_minimizer_counts),
            num_buckets: bucket_names.len() as u32,
            total_minimizers: total_entries,
            inverted: Some(InvertedManifest {
                format: ParquetShardFormat::Parquet,
                num_shards: shard_infos.len() as u32,
                total_entries,
                has_overlapping_shards: false,
                shards: shard_infos,
            }),
        };
        manifest.save(dir).unwrap();
    }

    /// All `(minimizer, bucket_id)` pairs in an index, grouped by bucket.
    fn all_pairs_by_bucket(dir: &Path) -> HashMap<u32, std::collections::HashSet<u64>> {
        let manifest = ParquetManifest::load(dir).unwrap();
        let mut out: HashMap<u32, std::collections::HashSet<u64>> = HashMap::new();
        for info in &manifest.inverted.unwrap().shards {
            let path = dir
                .join(files::INVERTED_DIR)
                .join(files::inverted_shard(info.shard_id));
            for (minimizer, bucket_id) in read_shard_pairs(&path).unwrap() {
                out.entry(bucket_id).or_default().insert(minimizer);
            }
        }
        out
    }

    fn write_delta(
        dir: &Path,
        target_bucket_id: u32,
        minimizers: &[u64],
    ) -> Vec<InvertedShardInfo> {
        super::super::create_index_directory(dir).unwrap();
        let pairs: Vec<(u64, u32)> = minimizers.iter().map(|&m| (m, target_bucket_id)).collect();
        let path = dir.join(files::INVERTED_DIR).join(files::inverted_shard(0));
        write_shard_from_pairs(&path, &pairs, &ParquetWriteOptions::default()).unwrap();
        vec![InvertedShardInfo {
            shard_id: 0,
            min_minimizer: minimizers.iter().copied().min().unwrap_or(0),
            max_minimizer: minimizers.iter().copied().max().unwrap_or(0),
            num_entries: minimizers.len() as u64,
        }]
    }

    #[test]
    fn apply_bucket_addition_adds_new_minimizers_without_touching_other_buckets() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("index.ryxdi");
        build_fixture_index(
            &index_dir,
            32,
            10,
            1,
            &[
                (1, "b1", vec![1, 2, 3]),
                (2, "b2", vec![10, 20, 30]),
                (3, "b3", vec![100, 200, 300]),
            ],
        );
        let before = all_pairs_by_bucket(&index_dir);

        let delta_dir = tmp.path().join("delta");
        let delta_shards = write_delta(&delta_dir, 2, &[15, 40]);

        let output_dir = tmp.path().join("output.ryxdi");
        let stats = apply_bucket_addition(
            &index_dir,
            2,
            &delta_dir,
            &delta_shards,
            &["new_genome.fasta".to_string()],
            &output_dir,
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert_eq!(
            stats.novel_minimizers, 2,
            "both new minimizers are genuinely new"
        );
        assert_eq!(stats.already_present, 0);

        let after = all_pairs_by_bucket(&output_dir);
        assert_eq!(
            after[&1], before[&1],
            "bucket 1 must be byte-for-byte unchanged"
        );
        assert_eq!(
            after[&3], before[&3],
            "bucket 3 must be byte-for-byte unchanged"
        );

        let mut expected_bucket2: std::collections::HashSet<u64> = before[&2].clone();
        expected_bucket2.extend([15, 40]);
        assert_eq!(
            after[&2], expected_bucket2,
            "bucket 2 is the union of old and new"
        );

        let (bucket_names, bucket_sources, _) = read_buckets_parquet(&output_dir).unwrap();
        assert_eq!(bucket_names.len(), 3, "no buckets were added or removed");
        assert!(bucket_sources[&2].contains(&"new_genome.fasta".to_string()));
        assert_eq!(bucket_sources[&1], vec!["orig_source".to_string()]);

        let manifest = ParquetManifest::load(&output_dir).unwrap();
        let actual_total: u64 = all_pairs_by_bucket(&output_dir)
            .values()
            .map(|s| s.len() as u64)
            .sum();
        assert_eq!(manifest.total_minimizers, actual_total);
    }

    #[test]
    fn apply_bucket_addition_dedups_against_existing_bucket_contents() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("index.ryxdi");
        build_fixture_index(
            &index_dir,
            32,
            10,
            1,
            &[(1, "b1", vec![1, 2, 3]), (2, "b2", vec![10, 20, 30])],
        );

        // Re-"discover" a file that only contains minimizers bucket 2 already has.
        let delta_dir = tmp.path().join("delta");
        let delta_shards = write_delta(&delta_dir, 2, &[10, 20, 30]);

        let output_dir = tmp.path().join("output.ryxdi");
        let stats = apply_bucket_addition(
            &index_dir,
            2,
            &delta_dir,
            &delta_shards,
            &["dup_genome.fasta".to_string()],
            &output_dir,
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert_eq!(
            stats.novel_minimizers, 0,
            "re-adding an already-present file must not report any new minimizers"
        );
        assert_eq!(stats.already_present, 3);

        let after = all_pairs_by_bucket(&output_dir);
        let expected: std::collections::HashSet<u64> = [10u64, 20, 30].into_iter().collect();
        assert_eq!(
            after[&2], expected,
            "index content is unchanged: dedup must not introduce duplicate pairs"
        );
    }

    #[test]
    fn apply_bucket_addition_keeps_has_overlapping_shards_false_for_single_bucket_index() {
        // Every shard is relevant when there's only one bucket, so nothing is
        // carried over untouched and the fast non-overlapping classify path
        // must remain available.
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("index.ryxdi");
        build_fixture_index(&index_dir, 32, 10, 1, &[(1, "only", vec![1, 2, 3])]);

        let delta_dir = tmp.path().join("delta");
        let delta_shards = write_delta(&delta_dir, 1, &[99]);

        let output_dir = tmp.path().join("output.ryxdi");
        let stats = apply_bucket_addition(
            &index_dir,
            1,
            &delta_dir,
            &delta_shards,
            &[],
            &output_dir,
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert_eq!(stats.shards_carried_over, 0);
        let manifest = ParquetManifest::load(&output_dir).unwrap();
        assert!(!manifest.inverted.unwrap().has_overlapping_shards);
    }

    #[test]
    fn in_place_matches_output_mode() {
        // Same inputs through both assembly paths must produce the same
        // index content, not merely "an index that also passes."
        let tmp = TempDir::new().unwrap();
        let buckets: &[(u32, &str, Vec<u64>)] = &[
            (1, "b1", vec![1, 2, 3]),
            (2, "b2", vec![10, 20, 30]),
            (3, "b3", vec![100, 200, 300]),
        ];

        let o_index_dir = tmp.path().join("o_index.ryxdi");
        build_fixture_index(&o_index_dir, 32, 10, 1, buckets);
        let o_delta_dir = tmp.path().join("o_delta");
        let o_delta_shards = write_delta(&o_delta_dir, 2, &[15, 40]);
        let output_dir = tmp.path().join("output.ryxdi");
        let o_stats = apply_bucket_addition(
            &o_index_dir,
            2,
            &o_delta_dir,
            &o_delta_shards,
            &["new_genome.fasta".to_string()],
            &output_dir,
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        let ip_index_dir = tmp.path().join("ip_index.ryxdi");
        build_fixture_index(&ip_index_dir, 32, 10, 1, buckets);
        let ip_delta_dir = tmp.path().join("ip_delta");
        let ip_delta_shards = write_delta(&ip_delta_dir, 2, &[15, 40]);
        let ip_stats = apply_bucket_addition_in_place(
            &ip_index_dir,
            2,
            &ip_delta_dir,
            &ip_delta_shards,
            &["new_genome.fasta".to_string()],
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert_eq!(o_stats.novel_minimizers, ip_stats.novel_minimizers);
        assert_eq!(o_stats.already_present, ip_stats.already_present);
        assert_eq!(o_stats.total_minimizers, ip_stats.total_minimizers);
        assert_eq!(
            all_pairs_by_bucket(&output_dir),
            all_pairs_by_bucket(&ip_index_dir),
            "in-place and -o modes must produce identical shard content"
        );

        let (o_names, o_sources, _) = read_buckets_parquet(&output_dir).unwrap();
        let (ip_names, ip_sources, _) = read_buckets_parquet(&ip_index_dir).unwrap();
        assert_eq!(o_names, ip_names);
        assert_eq!(o_sources, ip_sources);

        let o_manifest = ParquetManifest::load(&output_dir).unwrap();
        let ip_manifest = ParquetManifest::load(&ip_index_dir).unwrap();
        assert_eq!(o_manifest.source_hash, ip_manifest.source_hash);
        assert_eq!(
            o_manifest.inverted.unwrap().has_overlapping_shards,
            ip_manifest.inverted.unwrap().has_overlapping_shards
        );
    }

    #[test]
    fn in_place_deletes_superseded_shards_and_leaves_untouched_ones() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("index.ryxdi");
        build_fixture_index(
            &index_dir,
            32,
            10,
            1,
            &[(1, "b1", vec![1, 2, 3]), (2, "b2", vec![10, 20, 30])],
        );
        // bucket 1 is shard 0, bucket 2 is shard 1 (per build_fixture_index's
        // one-shard-per-bucket ordering).
        let superseded_shard = index_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(1));
        assert!(superseded_shard.exists());

        let delta_dir = tmp.path().join("delta");
        let delta_shards = write_delta(&delta_dir, 2, &[15]);
        apply_bucket_addition_in_place(
            &index_dir,
            2,
            &delta_dir,
            &delta_shards,
            &[],
            1024 * 1024,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert!(
            !superseded_shard.exists(),
            "the old bucket-2 shard must be removed after the commit"
        );
        let after = all_pairs_by_bucket(&index_dir);
        assert_eq!(after[&1], [1u64, 2, 3].into_iter().collect());
        assert_eq!(after[&2], [10u64, 20, 30, 15].into_iter().collect());
    }

    #[test]
    fn apply_bucket_addition_rejects_unknown_bucket() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("index.ryxdi");
        build_fixture_index(&index_dir, 32, 10, 1, &[(1, "b1", vec![1, 2, 3])]);

        let delta_dir = tmp.path().join("delta");
        let delta_shards = write_delta(&delta_dir, 99, &[5]);

        let output_dir = tmp.path().join("output.ryxdi");
        let result = apply_bucket_addition(
            &index_dir,
            99,
            &delta_dir,
            &delta_shards,
            &[],
            &output_dir,
            1024 * 1024,
            &ParquetWriteOptions::default(),
        );

        assert!(
            result.is_err(),
            "target bucket 99 does not exist in the index"
        );
    }

    #[test]
    fn bucket_id_range_reports_degenerate_range_for_single_bucket_shard() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        let pairs: Vec<(u64, u32)> = vec![(10, 7), (20, 7), (30, 7)];
        write_shard_from_pairs(&path, &pairs, &ParquetWriteOptions::default()).unwrap();

        assert_eq!(bucket_id_range(&path).unwrap(), (7, 7));
    }

    #[test]
    fn bucket_id_range_reports_full_span_for_mixed_bucket_shard() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        let pairs: Vec<(u64, u32)> = vec![(10, 2), (20, 5), (30, 9)];
        write_shard_from_pairs(&path, &pairs, &ParquetWriteOptions::default()).unwrap();

        assert_eq!(bucket_id_range(&path).unwrap(), (2, 9));
    }

    #[test]
    fn bucket_id_range_falls_back_to_full_range_when_stats_missing() {
        // A shard with zero row groups (no rows written) has no per-row-group
        // statistics to read; the fallback must claim the full range rather
        // than a bogus empty one.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        write_shard_from_pairs(&path, &[], &ParquetWriteOptions::default()).unwrap();

        assert_eq!(bucket_id_range(&path).unwrap(), (0, u32::MAX));
    }
}
