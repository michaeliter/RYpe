# Rype Architecture

Design rationale for cross-cutting decisions in Rype. For data structure layouts and field listings, read the code: `src/core.rs`, `src/classify.rs`, `src/indices/`. For constants and their values, see `src/constants.rs`.

## Why RY (purine/pyrimidine) encoding

A reduced 2-bit alphabet collapses purines and pyrimidines:

- Purines (A/G) → 1
- Pyrimidines (T/C) → 0
- Other bases (N, ambiguous) → invalid; resets k-mer extraction

This buys two properties:

1. **Mutation tolerance.** A↔G and T↔C substitutions don't break matches, which is desirable for noisy long reads and cross-strain comparisons.
2. **Density.** 64bp k-mers fit in a single `u64`, enabling fast hashing and compact storage. Reverse complement is `!kmer` in RY space.

The tradeoff is reduced specificity per k-mer. We compensate with longer k (default 64) and minimizer sketching.

## Why minimizers

A sliding window of size `w` over k-mers selects the minimum hash per window as the representative, with consecutive duplicates collapsed. Implemented with a monotonic deque for O(n) extraction.

This reduces the index size from ~|sequence| to ~|sequence|/w entries while preserving the property that homologous regions share minimizers with high probability. Typical `w` is 50–200.

## Why Parquet for indices

Indices are stored as `.ryxdi` directories:

- `manifest.toml` — k, w, salt, bucket metadata (human-readable)
- `buckets.parquet` — `(bucket_id, bucket_name, sources)`
- `inverted/shard.N.parquet` — sorted `(minimizer: u64, bucket_id: u32)` pairs

The shard layout enables the three properties we need:

- **Bounded memory during build** via streaming k-way merge of pre-sorted shards.
- **Bounded memory during classify** by loading one shard at a time (see `classify_batch_sharded_merge_join`).
- **Inspectability.** Manifests are human-readable; shards open in any Parquet tool.

DELTA_BINARY_PACKED encoding gives strong compression on sorted minimizer columns. Per-row-group bloom filters can reject I/O early when a batch's minimizers don't appear in a shard.

## Build → classify lifecycle

1. **Build** (`rype index create` or `from-config`): FASTA → minimizer extraction → sorted Parquet shards → manifest.
2. **Classify** (`rype classify run`): manifest loads instantly; for each batch of reads, extract query minimizers in parallel (rayon), then merge-join against each shard on disk.
3. **Negative filtering** (`-N` flag): a second index whose minimizers subtract from per-bucket scores. Indices must share `k`, `w`, and `salt` to be combinable.

The C API wraps steps 2–3 for FFI consumers.

## Updating an index

An index directory is otherwise immutable, but two commands rewrite one in place rather than requiring a from-scratch rebuild:

- **`rype index merge`**: combines multiple indices' buckets into one output index (optionally subtracting one index's minimizers from another first). It renumbers buckets and does not union same-named buckets across inputs — it's a structural combine, not an update to existing bucket contents.
- **`rype index bucket-update`**: adds new sequences to one *existing* bucket of one index — the "a database upgrade added new viral genomes, extend the viral bucket without rebuilding the microbial and eukaryotic ones" case. It relies on a property of `index from-config`-built multi-bucket indices, verified rather than assumed: every shard file belongs to exactly one bucket, which the Parquet `bucket_id` column's row-group statistics prove from the footer alone, with no need to read row data. Updating one bucket therefore only rewrites shards whose statistics overlap that bucket's id; every other shard is carried over untouched (hard-linked, falling back to a copy across filesystems). New minimizers are deduped against the bucket's existing contents by a streaming k-way merge, the same merge core `consolidate_shards_streaming` uses during a normal build.

  For indices where shards aren't bucket-exclusive (e.g. `index create`'s range-partitioned shards), the same footer-statistics check naturally degrades to "every shard is relevant," so the update becomes a full-index rewrite rather than silently missing minimizers.

  `-o <new.ryxdi>` (default) assembles the result in a fresh directory. `--in-place` instead writes new shards directly into the source index's own `inverted/` directory under previously-unused shard ids, then commits by atomically renaming a `manifest.toml.tmp` over `manifest.toml` — the single crash-safe commit point, matching how `ParquetManifest` guarantees callers never observe a partially-written manifest. Superseded shards are only deleted after that rename succeeds, so a crash mid-update leaves a few harmless orphan shard files next to a valid, unchanged index.

  Two known limitations are surfaced (via a warning), not silently papered over: per-file sequence-length statistics for the updated bucket go stale (the underlying per-file lengths aren't persisted, so median/stdev can't be recomputed from the new files alone), and orientation (`--orient`) isn't recorded in the manifest, so newly added sequences are extracted unoriented.
