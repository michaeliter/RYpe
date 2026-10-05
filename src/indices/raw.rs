//! Raw sidecar: a flat, uncompressed copy of an index's inverted shards, built
//! to be memory-mapped and searched in place.
//!
//! ```text
//! index.ryxdi/
//! └── raw/
//!     ├── manifest.toml
//!     ├── shard.{N}.minimizers.u64   # little-endian u64, non-decreasing
//!     └── shard.{N}.bucket_ids.u32   # little-endian u32, row-parallel to minimizers
//! ```
//!
//! Rows mirror the Parquet shard exactly (one row per `(minimizer, bucket_id)` pair,
//! same order), so a lookup against the sidecar returns exactly what the Parquet
//! loader returns. The files are headerless so the mmap base (page-aligned) is
//! always aligned for `u64`/`u32`.
//!
//! The sidecar trades disk (12 bytes/entry vs. ~4 for compressed Parquet) for
//! zero decode cost: opening is O(1) and the OS pages in only what lookups touch.

#[cfg(target_endian = "big")]
compile_error!("the raw sidecar format is little-endian; big-endian hosts are unsupported");

use crate::error::{Result, RypeError};
use crate::indices::parquet::hex_u64;
use crate::indices::parquet::merge::for_each_shard_batch;
use crate::indices::sharded::{ShardInfo, ShardedInvertedIndex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};

/// Sidecar directory name inside a `.ryxdi` index.
pub const RAW_DIR: &str = "raw";
/// Magic string identifying a raw sidecar manifest.
pub const RAW_FORMAT_MAGIC: &str = "RYPE_RAW_V1";
/// Raw sidecar format version. Increment on breaking changes.
pub const RAW_FORMAT_VERSION: u32 = 1;

/// How sidecar arrays are brought into memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RawLoad {
    /// Memory-map the files; the OS pages data in on demand.
    #[default]
    Mmap,
    /// Read the files fully into heap memory (for comparison against `Mmap`).
    Read,
}

/// Manifest stored at `raw/manifest.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawManifest {
    pub magic: String,
    pub format_version: u32,
    pub k: usize,
    pub w: usize,
    #[serde(with = "hex_u64")]
    pub salt: u64,
    /// Copied from the parent index; a mismatch means the sidecar is stale.
    #[serde(with = "hex_u64")]
    pub source_hash: u64,
    pub shards: Vec<RawShardInfo>,
}

/// Per-shard sidecar metadata. `min_minimizer`/`max_minimizer` are the first and
/// last values actually written (unlike the parent manifest's advisory bounds).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawShardInfo {
    pub shard_id: u32,
    pub num_entries: u64,
    /// Size of the source Parquet shard file. An O(1) staleness heuristic for
    /// rebuilds that change content but keep per-bucket counts (and thus
    /// `source_hash`); not a content hash.
    pub parquet_bytes: u64,
    #[serde(with = "hex_u64")]
    pub min_minimizer: u64,
    #[serde(with = "hex_u64")]
    pub max_minimizer: u64,
}

/// Write the raw sidecar for the index at `index_path`, replacing any existing one.
///
/// Streams each Parquet shard into flat files under `raw.tmp/`, verifying the
/// minimizer column is non-decreasing and the row count matches the manifest,
/// then renames `raw.tmp/` to `raw/` so a partial export is never picked up.
/// Column files and the manifest are fsynced before the rename.
pub fn export_raw(index_path: &Path) -> Result<RawManifest> {
    let sharded = ShardedInvertedIndex::open(index_path)?;
    let parent = sharded.manifest();

    let tmp_dir = index_path.join(format!("{}.tmp", RAW_DIR));
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)
            .map_err(|e| RypeError::io(&tmp_dir, "remove leftover raw.tmp", e))?;
    }
    fs::create_dir(&tmp_dir).map_err(|e| RypeError::io(&tmp_dir, "create raw.tmp", e))?;

    let written = parent
        .shards
        .iter()
        .map(|info| export_shard(&sharded.shard_path(info.shard_id), &tmp_dir, info))
        .collect::<Result<Vec<_>>>()
        .and_then(|shards| {
            let manifest = RawManifest {
                magic: RAW_FORMAT_MAGIC.to_string(),
                format_version: RAW_FORMAT_VERSION,
                k: parent.k,
                w: parent.w,
                salt: parent.salt,
                source_hash: parent.source_hash,
                shards,
            };
            manifest.save(&tmp_dir)?;
            Ok(manifest)
        });
    let manifest = match written {
        Ok(m) => m,
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }
    };

    // Move the old sidecar aside rather than deleting it first, so a failed
    // rename leaves the previous (valid) sidecar in place.
    let raw_dir = index_path.join(RAW_DIR);
    let old_dir = index_path.join(format!("{}.old", RAW_DIR));
    if old_dir.exists() {
        fs::remove_dir_all(&old_dir)
            .map_err(|e| RypeError::io(&old_dir, "remove leftover raw.old", e))?;
    }
    let had_old = raw_dir.exists();
    if had_old {
        fs::rename(&raw_dir, &old_dir).map_err(|e| RypeError::io(&raw_dir, "rename raw", e))?;
    }
    if let Err(e) = fs::rename(&tmp_dir, &raw_dir) {
        if had_old {
            let _ = fs::rename(&old_dir, &raw_dir);
        }
        return Err(RypeError::io(&raw_dir, "rename raw.tmp", e));
    }
    if had_old {
        if let Err(e) = fs::remove_dir_all(&old_dir) {
            log::warn!("could not remove {}: {}", old_dir.display(), e);
        }
    }
    Ok(manifest)
}

/// Copy one Parquet shard into `out_dir` as two flat column files.
fn export_shard(parquet_path: &Path, out_dir: &Path, info: &ShardInfo) -> Result<RawShardInfo> {
    let mins_path = raw_shard_path(out_dir, info.shard_id, MINIMIZERS_FILE);
    let bids_path = raw_shard_path(out_dir, info.shard_id, BUCKET_IDS_FILE);
    let create = |path: &Path| {
        File::create(path)
            .map(BufWriter::new)
            .map_err(|e| RypeError::io(path, "create raw sidecar column", e))
    };
    let mut mins_out = create(&mins_path)?;
    let mut bids_out = create(&bids_path)?;

    let mut num_entries = 0u64;
    let mut first: Option<u64> = None;
    let mut last: Option<u64> = None;
    for_each_shard_batch(parquet_path, |mins, bids| {
        // Lookups binary-search this column in place, so order is a hard invariant.
        let boundary_violation = matches!((last, mins.first()), (Some(p), Some(&m)) if m < p);
        if let Some(pos) = mins.windows(2).position(|w| w[0] > w[1]) {
            return Err(non_monotonic(parquet_path, num_entries + pos as u64 + 1));
        }
        if boundary_violation {
            return Err(non_monotonic(parquet_path, num_entries));
        }
        mins_out
            .write_all(as_bytes(mins))
            .map_err(|e| RypeError::io(&mins_path, "write raw sidecar column", e))?;
        bids_out
            .write_all(as_bytes(bids))
            .map_err(|e| RypeError::io(&bids_path, "write raw sidecar column", e))?;
        first = first.or(mins.first().copied());
        last = mins.last().copied().or(last);
        num_entries += mins.len() as u64;
        Ok(())
    })?;
    sync_writer(mins_out, &mins_path)?;
    sync_writer(bids_out, &bids_path)?;

    if num_entries != info.num_bucket_ids as u64 {
        return Err(RypeError::format(
            parquet_path,
            format!(
                "shard has {} rows but the index manifest records {}",
                num_entries, info.num_bucket_ids
            ),
        ));
    }
    Ok(RawShardInfo {
        shard_id: info.shard_id,
        num_entries,
        parquet_bytes: file_len(parquet_path)?,
        min_minimizer: first.unwrap_or(0),
        max_minimizer: last.unwrap_or(0),
    })
}

/// Flush buffered data and fsync, so a crash after export can't leave a
/// correctly-sized column file with unwritten (zeroed) pages.
fn sync_writer(writer: BufWriter<File>, path: &Path) -> Result<()> {
    let file = writer
        .into_inner()
        .map_err(|e| RypeError::io(path, "flush raw sidecar file", e.into_error()))?;
    file.sync_all()
        .map_err(|e| RypeError::io(path, "fsync raw sidecar file", e))
}

fn file_len(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)
        .map_err(|e| RypeError::io(path, "stat", e))?
        .len())
}

fn non_monotonic(path: &Path, row: u64) -> RypeError {
    RypeError::format(
        path,
        format!(
            "non-monotonic minimizer column at row {}: shard is not sorted by minimizer, \
             which the raw sidecar's in-place search requires",
            row
        ),
    )
}

/// An opened raw sidecar: one [`RawShard`] per parent shard.
pub struct RawIndex {
    shards: HashMap<u32, RawShard>,
}

impl std::fmt::Debug for RawIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawIndex")
            .field("num_shards", &self.shards.len())
            .finish()
    }
}

/// One shard's row-parallel `(minimizer, bucket_id)` arrays.
pub struct RawShard {
    minimizers: RawArray<u64>,
    bucket_ids: RawArray<u32>,
}

impl RawIndex {
    /// Open and validate the sidecar of `index` against its manifest.
    /// Validation is O(1) per shard (sizes, counts, first/last values) —
    /// sortedness was verified at export time.
    pub fn open(index: &ShardedInvertedIndex, load: RawLoad) -> Result<Self> {
        let index_path = index.base_path();
        let parent = index.manifest();
        let raw_dir = index_path.join(RAW_DIR);
        let manifest = match RawManifest::load(&raw_dir) {
            Err(RypeError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                return Err(RypeError::format(
                    &raw_dir,
                    format!(
                        "no raw sidecar found; create one with `rype index export-raw -i {}`",
                        index_path.display()
                    ),
                ));
            }
            other => other?,
        };
        let stale = |why: String| {
            RypeError::format(
                &raw_dir,
                format!(
                    "raw sidecar is stale ({}); re-run `rype index export-raw -i {}`",
                    why,
                    index_path.display()
                ),
            )
        };

        if (manifest.k, manifest.w, manifest.salt, manifest.source_hash)
            != (parent.k, parent.w, parent.salt, parent.source_hash)
        {
            return Err(stale("k/w/salt/source_hash differ from the index".into()));
        }
        // source_hash only covers per-bucket counts, so a re-shard of the same
        // content keeps it: per-shard row counts catch that.
        if manifest.shards.len() != parent.shards.len() {
            return Err(stale(format!(
                "{} shards vs {} in the index",
                manifest.shards.len(),
                parent.shards.len()
            )));
        }
        let by_id: HashMap<u32, &RawShardInfo> =
            manifest.shards.iter().map(|s| (s.shard_id, s)).collect();

        let mut shards = HashMap::with_capacity(parent.shards.len());
        for info in &parent.shards {
            let raw_info = by_id
                .get(&info.shard_id)
                .ok_or_else(|| stale(format!("shard {} missing", info.shard_id)))?;
            if raw_info.num_entries != info.num_bucket_ids as u64 {
                return Err(stale(format!(
                    "shard {} has {} rows vs {} in the index",
                    info.shard_id, raw_info.num_entries, info.num_bucket_ids
                )));
            }
            if raw_info.parquet_bytes != file_len(&index.shard_path(info.shard_id))? {
                return Err(stale(format!(
                    "Parquet shard {} changed size since export",
                    info.shard_id
                )));
            }
            let len = usize::try_from(raw_info.num_entries).map_err(|_| {
                RypeError::format(
                    &raw_dir,
                    "shard too large for this platform's address space",
                )
            })?;
            let mins_path = raw_shard_path(&raw_dir, info.shard_id, MINIMIZERS_FILE);
            let minimizers = RawArray::<u64>::open(&mins_path, len, load)?;
            let bucket_ids = RawArray::<u32>::open(
                &raw_shard_path(&raw_dir, info.shard_id, BUCKET_IDS_FILE),
                len,
                load,
            )?;
            let mins = minimizers.as_slice();
            if let (Some(&first), Some(&last)) = (mins.first(), mins.last()) {
                if (first, last) != (raw_info.min_minimizer, raw_info.max_minimizer) {
                    return Err(RypeError::format(
                        &mins_path,
                        "first/last minimizer differ from the sidecar manifest (corrupt file); \
                         re-run `rype index export-raw`",
                    ));
                }
            }
            shards.insert(
                info.shard_id,
                RawShard {
                    minimizers,
                    bucket_ids,
                },
            );
        }
        Ok(Self { shards })
    }

    /// The sidecar shard for `shard_id`, if present.
    pub fn shard(&self, shard_id: u32) -> Option<&RawShard> {
        self.shards.get(&shard_id)
    }
}

impl RawShard {
    /// Minimizer column, non-decreasing.
    pub fn minimizers(&self) -> &[u64] {
        self.minimizers.as_slice()
    }

    /// Bucket-id column, row-parallel to [`Self::minimizers`].
    pub fn bucket_ids(&self) -> &[u32] {
        self.bucket_ids.as_slice()
    }

    /// The `(minimizer, bucket_id)` rows whose minimizer occurs in `query`, in
    /// row order — exactly what the Parquet loader returns for the same shard
    /// and query. `query` must be sorted ascending (duplicates allowed).
    pub fn load_coo_for_query(&self, query: &[u64]) -> Result<Vec<(u64, u32)>> {
        if !query.windows(2).all(|w| w[0] <= w[1]) {
            return Err(RypeError::validation(
                "query_minimizers must be sorted in ascending order",
            ));
        }
        let t = std::time::Instant::now();
        let pairs = self.lookup_chunked(query, rayon::current_num_threads());
        crate::log_timing("raw_lookup: wall", t.elapsed().as_millis());
        Ok(pairs)
    }

    /// [`Self::load_coo_for_query`] split into `n_chunks` parallel query chunks.
    fn lookup_chunked(&self, query: &[u64], n_chunks: usize) -> Vec<(u64, u32)> {
        let (mins, bids) = (self.minimizers(), self.bucket_ids());
        let (Some(&lo), Some(&hi)) = (mins.first(), mins.last()) else {
            return Vec::new();
        };
        // Every query is offered to every shard; only values in range can match.
        let query = &query[query.partition_point(|&q| q < lo)..query.partition_point(|&q| q <= hi)];
        // Small queries aren't worth a rayon task per thread.
        let n_chunks =
            n_chunks.min(query.len().saturating_add(MIN_QUERY_PER_CHUNK - 1) / MIN_QUERY_PER_CHUNK);
        split_at_value_boundaries(query, n_chunks)
            .into_par_iter()
            .map(|chunk| {
                let r_lo = mins.partition_point(|&m| m < chunk[0]);
                let r_hi = mins.partition_point(|&m| m <= chunk[chunk.len() - 1]);
                intersect_rows(chunk, &mins[r_lo..r_hi], &bids[r_lo..r_hi])
            })
            .collect::<Vec<_>>()
            .concat()
    }
}

/// Minimum query values per parallel lookup chunk.
const MIN_QUERY_PER_CHUNK: usize = 4096;

/// Split sorted `q` into at most `n` contiguous non-empty chunks without
/// splitting a run of equal values (which would emit that run's rows twice).
fn split_at_value_boundaries(q: &[u64], n: usize) -> Vec<&[u64]> {
    if q.is_empty() {
        return Vec::new();
    }
    let n = n.clamp(1, q.len());
    // (len + n - 1) / n without div_ceil (MSRV 1.70); n <= len so no overflow.
    let step = (q.len() - 1) / n + 1;
    let mut chunks = Vec::with_capacity(n);
    let mut start = 0;
    for i in 1..n {
        let mut b = (step * i).clamp(start.max(1), q.len());
        if b < q.len() {
            let v = q[b - 1];
            b += q[b..].partition_point(|&x| x == v);
        }
        if b > start && b < q.len() {
            chunks.push(&q[start..b]);
            start = b;
        }
    }
    chunks.push(&q[start..]);
    chunks
}

/// Rows of (`mins`, `bids`) whose minimizer occurs in sorted `query`, in row
/// order. Gallops the shorter side over the longer one, so the cost is
/// O(short * log(long / short)) whichever side dominates. Unlike
/// `gallop_for_each`, this finds the *first* row of each value and emits the
/// whole run, since multi-bucket shards repeat a minimizer once per bucket.
fn intersect_rows(query: &[u64], mins: &[u64], bids: &[u32]) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    if query.len() <= mins.len() {
        let mut r = 0;
        let mut prev = None;
        for &q in query {
            if prev == Some(q) {
                continue;
            }
            prev = Some(q);
            r = gallop_lower_bound(mins, r, q);
            while r < mins.len() && mins[r] == q {
                out.push((q, bids[r]));
                r += 1;
            }
            if r == mins.len() {
                break;
            }
        }
    } else {
        let mut qi = 0;
        let mut r = 0;
        while r < mins.len() {
            let m = mins[r];
            let mut run_end = r + 1;
            while run_end < mins.len() && mins[run_end] == m {
                run_end += 1;
            }
            qi = gallop_lower_bound(query, qi, m);
            if qi == query.len() {
                break;
            }
            if query[qi] == m {
                out.extend((r..run_end).map(|i| (m, bids[i])));
            }
            r = run_end;
        }
    }
    out
}

/// First index `i >= from` with `v[i] >= target` (or `v.len()`), via
/// exponential then binary search: O(log(i - from)).
fn gallop_lower_bound(v: &[u64], from: usize, target: u64) -> usize {
    if from >= v.len() || v[from] >= target {
        return from;
    }
    // Invariant: v[lo] < target.
    let mut lo = from;
    let mut step = 1;
    loop {
        let probe = lo + step;
        if probe >= v.len() || v[probe] >= target {
            let end = probe.min(v.len());
            return lo + 1 + v[lo + 1..end].partition_point(|&x| x < target);
        }
        lo = probe;
        step *= 2;
    }
}

impl RawManifest {
    fn save(&self, raw_dir: &Path) -> Result<()> {
        let path = raw_dir.join(MANIFEST_FILE);
        let text = toml::to_string_pretty(self)
            .map_err(|e| RypeError::encoding(format!("serialize raw manifest: {}", e)))?;
        let mut out = BufWriter::new(
            File::create(&path).map_err(|e| RypeError::io(&path, "create raw manifest", e))?,
        );
        out.write_all(text.as_bytes())
            .map_err(|e| RypeError::io(&path, "write raw manifest", e))?;
        sync_writer(out, &path)
    }

    fn load(raw_dir: &Path) -> Result<Self> {
        let path = raw_dir.join(MANIFEST_FILE);
        let text =
            fs::read_to_string(&path).map_err(|e| RypeError::io(&path, "read raw manifest", e))?;
        let manifest: Self = toml::from_str(&text)
            .map_err(|e| RypeError::format(&path, format!("parse raw manifest: {}", e)))?;
        if manifest.magic != RAW_FORMAT_MAGIC || manifest.format_version != RAW_FORMAT_VERSION {
            return Err(RypeError::format(
                &path,
                format!(
                    "unsupported raw sidecar: magic '{}' version {} (expected '{}' version {})",
                    manifest.magic, manifest.format_version, RAW_FORMAT_MAGIC, RAW_FORMAT_VERSION
                ),
            ));
        }
        Ok(manifest)
    }
}

const MANIFEST_FILE: &str = "manifest.toml";
const MINIMIZERS_FILE: &str = "minimizers.u64";
const BUCKET_IDS_FILE: &str = "bucket_ids.u32";

fn raw_shard_path(raw_dir: &Path, shard_id: u32, column: &str) -> PathBuf {
    raw_dir.join(format!("shard.{}.{}", shard_id, column))
}

/// Element types of sidecar column files: plain integers with no padding and
/// no invalid bit patterns, stored in host (= little-endian) byte order.
trait RawElem: Copy + Default + 'static {}
impl RawElem for u64 {}
impl RawElem for u32 {}

fn as_bytes<T: RawElem>(v: &[T]) -> &[u8] {
    // SAFETY: T is u64 or u32 (no padding); every byte is initialized and the
    // byte length is exactly size_of_val(v).
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T: RawElem>(v: &mut [T]) -> &mut [u8] {
    // SAFETY: as in `as_bytes`; additionally any byte pattern is a valid T.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// A column file either memory-mapped or read into the heap.
enum RawArray<T> {
    Mapped(memmap2::Mmap, PhantomData<T>),
    Owned(Vec<T>),
}

impl<T: RawElem> RawArray<T> {
    fn open(path: &Path, len: usize, load: RawLoad) -> Result<Self> {
        let mut file =
            File::open(path).map_err(|e| RypeError::io(path, "open raw sidecar column", e))?;
        let expected = (len as u64)
            .checked_mul(size_of::<T>() as u64)
            .ok_or_else(|| RypeError::format(path, "column size overflows u64"))?;
        let actual = file
            .metadata()
            .map_err(|e| RypeError::io(path, "stat raw sidecar column", e))?
            .len();
        if actual != expected {
            return Err(RypeError::format(
                path,
                format!(
                    "file size {} bytes does not match {} entries x {} bytes; the sidecar is \
                     truncated or corrupt — re-run `rype index export-raw`",
                    actual,
                    len,
                    size_of::<T>()
                ),
            ));
        }
        if len == 0 {
            // Mapping an empty file is not portable.
            return Ok(Self::Owned(Vec::new()));
        }
        match load {
            RawLoad::Mmap => {
                // SAFETY: sidecar files are treated as immutable. `export_raw`
                // replaces them via rename, which leaves existing mappings of the
                // old files intact; truncating them in place while mapped is
                // unsupported (it would fault on access).
                let map = unsafe { memmap2::Mmap::map(&file) }
                    .map_err(|e| RypeError::io(path, "mmap raw sidecar column", e))?;
                // SAFETY: any bit pattern is a valid T; we only inspect alignment here.
                let (head, _, tail) = unsafe { map.align_to::<T>() };
                if !head.is_empty() || !tail.is_empty() {
                    return Err(RypeError::format(
                        path,
                        "mapping is not aligned for its type",
                    ));
                }
                Ok(Self::Mapped(map, PhantomData))
            }
            RawLoad::Read => {
                let mut v = vec![T::default(); len];
                file.read_exact(as_bytes_mut(&mut v))
                    .map_err(|e| RypeError::io(path, "read raw sidecar column", e))?;
                Ok(Self::Owned(v))
            }
        }
    }

    fn as_slice(&self) -> &[T] {
        match self {
            Self::Mapped(map, _) => {
                // SAFETY: alignment and exact length were verified in `open`, the
                // mapping is immutable for its lifetime, and any bit pattern is a valid T.
                let (_, mid, _) = unsafe { map.align_to::<T>() };
                mid
            }
            Self::Owned(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::merge::read_shard_pairs;
    use crate::indices::parquet::{create_parquet_inverted_index, BucketData, ParquetWriteOptions};
    use crate::indices::sharded::ShardManifest;
    use tempfile::TempDir;

    /// Minimizers above i64::MAX are realistic (k=64 RY values reach 0xAAAA...)
    /// and exercise the u64-as-INT64 Parquet path.
    const HIGH: u64 = 0xAAAA_AAAA_AAAA_AAAA;

    /// Deterministic, well-spread sorted unique minimizers for one bucket.
    fn bucket_minimizers(seed: u64, n: usize) -> Vec<u64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut v: Vec<u64> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Three buckets that share some minimizers (so shards contain duplicate
    /// minimizer values with different bucket ids) plus edge values.
    fn test_buckets() -> Vec<BucketData> {
        let shared = bucket_minimizers(99, 2_000);
        let mut buckets = Vec::new();
        for (id, seed) in [(2u32, 1u64), (5, 2), (7, 3)] {
            let mut mins = bucket_minimizers(seed, 100_000);
            mins.extend_from_slice(&shared);
            mins.extend_from_slice(&[0, 1, 1 << 63, HIGH]);
            mins.sort_unstable();
            mins.dedup();
            buckets.push(BucketData {
                bucket_id: id,
                bucket_name: format!("b{}", id),
                sources: vec![format!("src{}", id)],
                minimizers: mins,
            });
        }
        buckets
    }

    /// Build a multi-shard, multi-row-group index (MIN_SHARD_BYTES = 1MB).
    fn build_index(path: &Path, buckets: Vec<BucketData>, max_shard_bytes: usize) {
        let opts = ParquetWriteOptions {
            row_group_size: 5_000,
            ..Default::default()
        };
        create_parquet_inverted_index(
            path,
            buckets,
            64,
            20,
            0x5555,
            Some(max_shard_bytes),
            Some(&opts),
            None,
        )
        .unwrap();
    }

    fn open_raw(path: &Path, load: RawLoad) -> Result<RawIndex> {
        RawIndex::open(&ShardedInvertedIndex::open(path).unwrap(), load)
    }

    fn parent(path: &Path) -> ShardManifest {
        ShardedInvertedIndex::open(path).unwrap().manifest().clone()
    }

    #[test]
    fn export_then_open_reproduces_every_parquet_row_in_order() {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        let manifest = parent(&idx);
        // Preconditions: the test must exercise multiple shards and duplicate
        // minimizers within a shard, or row-for-row equality proves little.
        assert!(manifest.shards.len() > 1, "need >1 shard");

        export_raw(&idx).unwrap();

        let mut saw_duplicate_minimizer = false;
        for load in [RawLoad::Mmap, RawLoad::Read] {
            let raw = open_raw(&idx, load).unwrap();
            for info in &manifest.shards {
                let expected =
                    read_shard_pairs(&ShardManifest::shard_path_parquet(&idx, info.shard_id))
                        .unwrap();
                let shard = raw.shard(info.shard_id).unwrap();
                let got: Vec<(u64, u32)> = shard
                    .minimizers()
                    .iter()
                    .copied()
                    .zip(shard.bucket_ids().iter().copied())
                    .collect();
                assert_eq!(got, expected, "shard {} ({:?})", info.shard_id, load);
                saw_duplicate_minimizer |= shard.minimizers().windows(2).any(|w| w[0] == w[1]);
            }
        }
        assert!(
            saw_duplicate_minimizer,
            "fixture must contain shared minimizers"
        );
    }

    #[test]
    fn export_records_actual_first_and_last_values() {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);

        let written = export_raw(&idx).unwrap();
        let raw = open_raw(&idx, RawLoad::Mmap).unwrap();
        for info in &written.shards {
            let mins = raw.shard(info.shard_id).unwrap().minimizers();
            assert_eq!(info.num_entries as usize, mins.len());
            assert_eq!(info.min_minimizer, mins[0]);
            assert_eq!(info.max_minimizer, *mins.last().unwrap());
        }
        let global_max = written.shards.iter().map(|s| s.max_minimizer).max();
        let true_max = test_buckets()
            .iter()
            .flat_map(|b| b.minimizers.iter().copied())
            .max();
        assert!(true_max.unwrap() > i64::MAX as u64, "precondition");
        assert_eq!(global_max, true_max, "values above i64::MAX must survive");
    }

    #[test]
    fn open_without_sidecar_errors_with_export_hint() {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);

        let err = open_raw(&idx, RawLoad::Mmap).unwrap_err();
        assert!(err.to_string().contains("export-raw"), "got: {err}");
    }

    #[test]
    fn open_rejects_sidecar_from_a_rebuilt_index() {
        // Rebuilding with different content changes source_hash: the old
        // sidecar no longer describes the index and must not be used.
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        export_raw(&idx).unwrap();

        let mut buckets = test_buckets();
        buckets[0].minimizers.truncate(1_000);
        build_index(&idx, buckets, 1 << 20);

        let err = open_raw(&idx, RawLoad::Mmap).unwrap_err();
        assert!(err.to_string().contains("stale"), "got: {err}");

        // Re-exporting repairs it.
        export_raw(&idx).unwrap();
        open_raw(&idx, RawLoad::Mmap).unwrap();
    }

    #[test]
    fn open_rejects_sidecar_from_a_resharded_index() {
        // Same content (so same source_hash) but a different shard split:
        // shard N of the sidecar no longer corresponds to Parquet shard N.
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        export_raw(&idx).unwrap();
        let before = parent(&idx);

        build_index(&idx, test_buckets(), 2 << 20);
        let after = parent(&idx);
        assert_eq!(before.source_hash, after.source_hash, "precondition");
        assert_ne!(before.shards.len(), after.shards.len(), "precondition");

        let err = open_raw(&idx, RawLoad::Mmap).unwrap_err();
        assert!(err.to_string().contains("stale"), "got: {err}");
    }

    #[test]
    fn open_rejects_sidecar_when_content_changes_but_counts_do_not() {
        // Same per-bucket counts => same source_hash and row counts; only the
        // Parquet shard's size reveals the rebuild.
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        let single_shard = 1 << 40;
        build_index(&idx, test_buckets(), single_shard);
        export_raw(&idx).unwrap();
        let before = parent(&idx);
        let shard0 = ShardManifest::shard_path_parquet(&idx, 0);
        let size_before = std::fs::metadata(&shard0).unwrap().len();

        let mut buckets = test_buckets();
        for m in buckets[0].minimizers.iter_mut().skip(10).step_by(7) {
            *m ^= 1 << 40; // different values, same count
        }
        buckets[0].minimizers.sort_unstable();
        buckets[0].minimizers.dedup();
        assert_eq!(
            buckets[0].minimizers.len(),
            test_buckets()[0].minimizers.len(),
            "precondition"
        );
        build_index(&idx, buckets, single_shard);
        let after = parent(&idx);
        assert_eq!(before.source_hash, after.source_hash, "precondition");
        assert_eq!(before.shards.len(), after.shards.len(), "precondition");
        assert_ne!(
            size_before,
            std::fs::metadata(&shard0).unwrap().len(),
            "precondition"
        );

        let err = open_raw(&idx, RawLoad::Mmap).unwrap_err();
        assert!(err.to_string().contains("stale"), "got: {err}");
    }

    #[test]
    fn open_rejects_parameter_mismatch() {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        export_raw(&idx).unwrap();

        let manifest_path = idx.join(RAW_DIR).join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&manifest_path).unwrap();
        assert!(text.contains("w = 20"), "precondition");
        std::fs::write(&manifest_path, text.replace("w = 20", "w = 21")).unwrap();
        let err = open_raw(&idx, RawLoad::Mmap).unwrap_err();
        assert!(err.to_string().contains("stale"), "got: {err}");
    }

    #[test]
    fn open_rejects_truncated_column_file() {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        export_raw(&idx).unwrap();
        let manifest = parent(&idx);

        for column in ["minimizers.u64", "bucket_ids.u32"] {
            let tmp2 = TempDir::new().unwrap();
            let copy = tmp2.path().join("idx.ryxdi");
            copy_dir(&idx, &copy);
            let path = raw_shard_path(&copy.join(RAW_DIR), manifest.shards[0].shard_id, column);
            let len = std::fs::metadata(&path).unwrap().len();
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(len - 4).unwrap();

            for load in [RawLoad::Mmap, RawLoad::Read] {
                let err = open_raw(&copy, load).unwrap_err();
                assert!(err.to_string().contains("size"), "{column}: got: {err}");
            }
        }
    }

    #[test]
    fn export_rejects_unsorted_shard() {
        use arrow::array::{UInt32Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, test_buckets(), 1 << 20);
        let manifest = parent(&idx);
        let shard_path = ShardManifest::shard_path_parquet(&idx, manifest.shards[0].shard_id);

        // Same row count, two rows swapped: an externally produced bad shard.
        let mut pairs = read_shard_pairs(&shard_path).unwrap();
        let mid = pairs.len() / 2;
        assert_ne!(pairs[mid].0, pairs[mid + 1].0, "precondition");
        pairs.swap(mid, mid + 1);
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(pairs.iter().map(|p| p.0))),
                Arc::new(UInt32Array::from_iter_values(pairs.iter().map(|p| p.1))),
            ],
        )
        .unwrap();
        let mut w = ArrowWriter::try_new(std::fs::File::create(&shard_path).unwrap(), schema, None)
            .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let err = export_raw(&idx).unwrap_err();
        assert!(err.to_string().contains("non-monotonic"), "got: {err}");
        assert!(
            !idx.join(RAW_DIR).exists(),
            "failed export must not leave a sidecar"
        );
    }

    // ---------------------------------------------------------------------
    // Lookup equivalence (Phase 2): raw lookups must return exactly the
    // Parquet loader's rows, or classification results would diverge.
    // ---------------------------------------------------------------------

    /// Sorted query: `n_hits` values sampled from `pool` plus `n_misses`
    /// random values, optionally with every value duplicated.
    fn make_query(seed: u64, pool: &[u64], n_hits: usize, n_misses: usize, dup: bool) -> Vec<u64> {
        let mut q: Vec<u64> = bucket_minimizers(seed, n_misses);
        let r = bucket_minimizers(seed ^ 0xDEAD, n_hits);
        q.extend(r.iter().map(|&x| pool[(x % pool.len() as u64) as usize]));
        if dup {
            q.extend(q.clone());
        }
        q.sort_unstable();
        q
    }

    fn exported_index(buckets: Vec<BucketData>, max_shard_bytes: usize) -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let idx = tmp.path().join("idx.ryxdi");
        build_index(&idx, buckets, max_shard_bytes);
        export_raw(&idx).unwrap();
        (tmp, idx)
    }

    /// All query shapes the lookup must handle, built from the index's own values.
    fn query_cases(all: &[u64]) -> Vec<(&'static str, Vec<u64>)> {
        let below = all[0].saturating_sub(1);
        let above = all.last().unwrap().saturating_add(1);
        let mut out_of_range = vec![above];
        if all[0] > 0 {
            out_of_range.insert(0, below);
        }
        vec![
            ("empty", vec![]),
            ("single hit", vec![all[all.len() / 2]]),
            ("single miss", vec![all[all.len() / 2] ^ 1 << 50]),
            ("out of range", out_of_range),
            ("sparse", make_query(11, all, 50, 50, false)),
            ("medium", make_query(12, all, 20_000, 20_000, false)),
            (
                "dense with duplicates",
                make_query(13, all, 150_000, 50_000, true),
            ),
            ("every index value", {
                let mut v = all.to_vec();
                v.dedup();
                v
            }),
        ]
    }

    fn all_minimizers(idx: &Path) -> Vec<u64> {
        let sharded = ShardedInvertedIndex::open(idx).unwrap();
        let mut all: Vec<u64> = sharded
            .manifest()
            .shards
            .iter()
            .flat_map(|s| read_shard_pairs(&sharded.shard_path(s.shard_id)).unwrap())
            .map(|p| p.0)
            .collect();
        all.sort_unstable();
        all
    }

    #[test]
    fn raw_lookup_matches_parquet_loader_exactly() {
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 20);
        let parquet = ShardedInvertedIndex::open(&idx).unwrap();
        assert!(parquet.manifest().shards.len() > 1, "precondition");
        for (name, q) in query_cases(&all_minimizers(&idx)) {
            for load in [RawLoad::Mmap, RawLoad::Read] {
                let raw = open_raw(&idx, load).unwrap();
                for info in &parquet.manifest().shards {
                    let expected = parquet
                        .load_shard_coo_for_query(info.shard_id, &q, None)
                        .unwrap();
                    let got = raw
                        .shard(info.shard_id)
                        .unwrap()
                        .load_coo_for_query(&q)
                        .unwrap();
                    assert_eq!(got, expected, "{name}, shard {}, {load:?}", info.shard_id);
                }
            }
        }
    }

    #[test]
    fn chunked_lookup_matches_single_chunk_for_any_chunk_count() {
        // Chunk boundaries must never split a run of equal query values (or
        // the run's rows would be emitted twice) nor drop rows at the edges.
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 40);
        let raw = open_raw(&idx, RawLoad::Mmap).unwrap();
        let shard = raw.shard(0).unwrap();
        let q = make_query(21, &all_minimizers(&idx), 5_000, 5_000, true);
        let reference = shard.lookup_chunked(&q, 1);
        assert!(!reference.is_empty(), "precondition");
        for n in [2, 3, 7, 64, q.len(), q.len() + 5] {
            assert_eq!(shard.lookup_chunked(&q, n), reference, "n_chunks = {n}");
        }

        // One value repeated across every would-be boundary.
        let v = q[q.len() / 2];
        let mut runs = vec![v; 50_000];
        runs.extend_from_slice(&q[..100]);
        runs.sort_unstable();
        let reference = shard.lookup_chunked(&runs, 1);
        assert!(reference.iter().any(|p| p.0 == v), "precondition");
        for n in [2, 5, 13] {
            assert_eq!(
                shard.lookup_chunked(&runs, n),
                reference,
                "long run, n = {n}"
            );
        }
    }

    #[test]
    fn lookup_handles_query_much_larger_than_shard() {
        // Every query is passed to every shard, so a small shard can face a
        // query far larger than itself; results must still match Parquet.
        let small = vec![BucketData {
            bucket_id: 3,
            bucket_name: "small".into(),
            sources: vec!["s".into()],
            minimizers: bucket_minimizers(5, 500),
        }];
        let (_tmp, idx) = exported_index(small, 1 << 40);
        let parquet = ShardedInvertedIndex::open(&idx).unwrap();
        let raw = open_raw(&idx, RawLoad::Mmap).unwrap();
        let q = make_query(31, &all_minimizers(&idx), 300, 200_000, false);
        let expected = parquet.load_shard_coo_for_query(0, &q, None).unwrap();
        assert!(!expected.is_empty(), "precondition");
        for n in [1, 4] {
            assert_eq!(raw.shard(0).unwrap().lookup_chunked(&q, n), expected);
        }
    }

    #[test]
    fn lookup_rejects_unsorted_query() {
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 40);
        let raw = open_raw(&idx, RawLoad::Mmap).unwrap();
        let err = raw
            .shard(0)
            .unwrap()
            .load_coo_for_query(&[5, 3])
            .unwrap_err();
        assert!(err.to_string().contains("sorted"), "got: {err}");
    }

    #[test]
    fn attached_sidecar_serves_both_coo_and_csr_loaders() {
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 20);
        let q = make_query(41, &all_minimizers(&idx), 30_000, 30_000, false);
        let parquet = ShardedInvertedIndex::open(&idx).unwrap();
        let mut with_raw = ShardedInvertedIndex::open(&idx).unwrap();
        with_raw.attach_raw(RawLoad::Mmap).unwrap();
        for info in &parquet.manifest().shards {
            let id = info.shard_id;
            assert_eq!(
                with_raw.load_shard_coo_for_query(id, &q, None).unwrap(),
                parquet.load_shard_coo_for_query(id, &q, None).unwrap()
            );
            let (a, b) = (
                with_raw.load_shard_for_query(id, &q, None).unwrap(),
                parquet.load_shard_for_query(id, &q, None).unwrap(),
            );
            assert_eq!(
                (&a.minimizers, &a.offsets, &a.bucket_ids),
                (&b.minimizers, &b.offsets, &b.bucket_ids)
            );
            assert_eq!(
                (a.k, a.w, a.salt, a.source_hash),
                (b.k, b.w, b.salt, b.source_hash)
            );
        }
    }

    #[test]
    fn attached_sidecar_is_actually_read() {
        // Equality with Parquet can't distinguish "used the sidecar" from
        // "silently fell back to Parquet". Rewrite the sidecar's bucket ids
        // (sizes stay valid) and check both loaders report the rewritten ids.
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 40);
        let bids = raw_shard_path(&idx.join(RAW_DIR), 0, BUCKET_IDS_FILE);
        let n = std::fs::metadata(&bids).unwrap().len() as usize / 4;
        std::fs::write(
            &bids,
            vec![99u32; n]
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();

        let mut sharded = ShardedInvertedIndex::open(&idx).unwrap();
        sharded.attach_raw(RawLoad::Mmap).unwrap();
        let q = make_query(51, &all_minimizers(&idx), 1_000, 0, false);
        let coo = sharded.load_shard_coo_for_query(0, &q, None).unwrap();
        assert!(!coo.is_empty() && coo.iter().all(|&(_, b)| b == 99));
        let csr = sharded.load_shard_for_query(0, &q, None).unwrap();
        assert!(!csr.bucket_ids.is_empty() && csr.bucket_ids.iter().all(|&b| b == 99));
    }

    #[test]
    fn classification_with_sidecar_matches_parquet_on_overlapping_multi_bucket_shards() {
        // End-to-end through the shard loop: cross-shard dedup, accumulation
        // and scoring must see identical rows whichever store served them.
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 20);
        let parquet = ShardedInvertedIndex::open(&idx).unwrap();
        assert!(parquet.manifest().has_overlapping_shards, "precondition");
        assert!(parquet.manifest().shards.len() > 1, "precondition");
        let all = all_minimizers(&idx);

        // Reads with varying hit fractions on each strand.
        let extracted: Vec<(Vec<u64>, Vec<u64>)> = (0..300u64)
            .map(|i| {
                let hits = (i % 40) as usize;
                let mut fwd = make_query(100 + i, &all, hits, 40 - hits, false);
                fwd.dedup();
                let mut rc = make_query(1000 + i, &all, (i % 7) as usize, 10, false);
                rc.dedup();
                (fwd, rc)
            })
            .collect();
        let ids: Vec<i64> = (0..extracted.len() as i64).collect();

        let mut with_raw = ShardedInvertedIndex::open(&idx).unwrap();
        with_raw.attach_raw(RawLoad::Mmap).unwrap();
        let mut prev_len = usize::MAX;
        for threshold in [0.0, 0.1, 0.25] {
            let sorted = |sharded: &ShardedInvertedIndex| {
                let mut hits = crate::classify::classify_from_extracted_minimizers(
                    sharded, &extracted, &ids, threshold, None,
                )
                .unwrap();
                hits.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));
                hits
            };
            let expected = sorted(&parquet);
            // Each threshold must keep some hits and drop some (vs. the lower one).
            assert!(
                !expected.is_empty() && expected.len() < prev_len,
                "t={threshold}"
            );
            prev_len = expected.len();
            let got = sorted(&with_raw);
            assert_eq!(got.len(), expected.len(), "t={threshold}");
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!((g.query_id, g.bucket_id), (e.query_id, e.bucket_id));
                assert_eq!(g.score.to_bits(), e.score.to_bits(), "t={threshold}");
            }
        }
    }

    #[test]
    fn parallel_rg_classification_rejects_attached_sidecar() {
        // Row-group parallelism decodes Parquet directly; with a sidecar
        // attached it must fail loudly rather than silently bypass it.
        let (_tmp, idx) = exported_index(test_buckets(), 1 << 40);
        let mut sharded = ShardedInvertedIndex::open(&idx).unwrap();
        sharded.attach_raw(RawLoad::Mmap).unwrap();
        let mut fwd = make_query(61, &all_minimizers(&idx), 200, 0, false);
        fwd.dedup();
        let err = crate::classify::classify_from_extracted_minimizers_parallel_rg(
            &sharded,
            &[(fwd, Vec::new())],
            &[7],
            0.0,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("raw sidecar"), "got: {err}");
    }

    fn copy_dir(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).unwrap();
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let target = dst.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }
}
