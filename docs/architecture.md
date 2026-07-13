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

## Cross-bucket minimizer dedup (`--dedup-cross-bucket`)

Multi-bucket indices can accumulate minimizers shared across many buckets — e.g. a conserved core genome shared by closely related strains. These add classification noise: a read landing entirely on shared content scores equally against every bucket that contains it, rather than identifying its true source.

`--dedup-cross-bucket` (opt-in, `--dedup-cross-bucket-min N`, default `N=2`) removes any minimizer present in `N` or more buckets from *all* buckets that contain it, at build time. `rype index create` does this in a single pass (all buckets already resident in memory). `rype index from-config` (multi-bucket) does it in two passes — extract once to count bucket membership, extract again to filter and write — to preserve the streaming path's bounded-memory design rather than holding every bucket's minimizers in memory at once. The counting pass is order-independent, so it's safe to run before `--orient`'s (also order-independent) orientation choice.

**What it actually buys you, empirically** (tested against real WoL2 genomes — 5 *E. coli* strains plus 4 distantly related taxa): the benefit and the cost are both concentrated in whichever buckets actually share content.

- For genuinely distinct organisms (different genus/family), baseline classification is already ~100% accurate and dedup changes essentially nothing — there's no shared content to remove (observed: <40 minimizers removed out of hundreds of thousands per bucket).
- For closely related genomes (same species, different strains), dedup substantially reduces cross-bucket false-positive scoring (in one measurement, off-target hit rate on sibling strains dropped from ~90% to ~35% at the default threshold) — but at a real sensitivity cost, since the same shared minimizers that caused confusion also contributed to some reads' correct self-identification. It is **not** equivalent to keeping the baseline index and doing best-hit tie-breaking: it's a global, count-based filter applied uniformly to the reference data, not a per-read decision, so it doesn't precisely target only the reads that were actually ambiguous.
- This feature was designed for distinguishing different organisms, not resolving fine-grained strain-level identity — don't expect it to cleanly separate near-identical genomes.

Classification against a deduped index is also measurably faster and lighter, roughly proportional to how much smaller the index gets (fewer minimizers to decode and merge-join against), independent of any accuracy effect.

## Build → classify lifecycle

1. **Build** (`rype index create` or `from-config`): FASTA → minimizer extraction → sorted Parquet shards → manifest.
2. **Classify** (`rype classify run`): manifest loads instantly; for each batch of reads, extract query minimizers in parallel (rayon), then merge-join against each shard on disk.
3. **Negative filtering** (`-N` flag): a second index whose minimizers subtract from per-bucket scores. Indices must share `k`, `w`, and `salt` to be combinable.

The C API wraps steps 2–3 for FFI consumers.
