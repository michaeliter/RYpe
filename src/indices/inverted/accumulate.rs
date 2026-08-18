//! Cross-batch query accumulation for single-pass classification.
//!
//! `run_log_ratio`'s `DeferredDenomBuffer` (`src/commands/helpers/deferred_denom.rs`)
//! established the pattern this module generalizes: accumulate reads as flat COO
//! `(minimizer, packed_read_id)` entries across many input batches, then drain into
//! one sorted `QueryInvertedIndex` for a single index scan. The reason to flatten
//! immediately rather than hold `Vec<(Vec<u64>, Vec<u64>)>` per read (as
//! `QueryInvertedIndex::build` does) is residency: holding both the raw per-read
//! minimizer vectors AND the flattened, sorted entries at once roughly doubles peak
//! memory for no benefit. `QueryAccumulator::push` drops each read's vectors the
//! moment they are flattened, so only one representation is ever resident.
//!
//! Generic over `M`, the per-read metadata payload, so both callers can share this
//! type: the CLI carries `String` headers (for `format_classification_results`,
//! which indexes output rows by position), and the Arrow/C-API path carries `i64`
//! query ids.

use super::query::QueryInvertedIndex;

/// Accumulates extracted minimizers across many input batches as flat COO entries,
/// flushing when a caller-supplied byte budget (not a read count) is reached.
///
/// # Invariants
/// - `entries` is unsorted while accumulating; `drain()` sorts it.
/// - `fwd_counts.len() == rc_counts.len() == meta.len()` == reads pushed since the
///   last `drain()`.
pub struct QueryAccumulator<M> {
    entries: Vec<(u64, u32)>,
    fwd_counts: Vec<u32>,
    rc_counts: Vec<u32>,
    meta: Vec<M>,
    byte_budget: usize,
    max_reads: Option<usize>,
    accumulator_bytes_per_read: usize,
}

impl<M> QueryAccumulator<M> {
    /// Create an accumulator that signals `should_flush()` once accumulated data
    /// reaches approximately `byte_budget` bytes.
    pub fn with_budget(byte_budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            fwd_counts: Vec::new(),
            rc_counts: Vec::new(),
            meta: Vec::new(),
            byte_budget,
            max_reads: None,
            accumulator_bytes_per_read: 0,
        }
    }

    /// Additionally flush once `max_reads` reads have been pushed, regardless of
    /// byte budget. Used to preserve exact read-count semantics when a caller
    /// (e.g. `--batch-size`) asks for a specific number of reads per pass rather
    /// than a memory-driven one.
    pub fn with_max_reads(mut self, max_reads: Option<usize>) -> Self {
        self.max_reads = max_reads;
        self
    }

    /// Reserve `bytes_per_read` of the byte budget per accumulated read for the
    /// `HitAccumulator` a classification pass will allocate over this group —
    /// see `memory::estimate_accumulator_bytes_per_read`. Entries and counts are
    /// bounded by minimizer count, but the accumulator scales with *read* count
    /// (up to ~1.3 KB/read on a 160-bucket dense index), so a pass with few but
    /// long reads can be accumulator-bound rather than entry-bound. Without this,
    /// `should_flush()` only sees entry/count bytes and a large pass can exceed
    /// its budget entirely in accumulator allocation once classification starts.
    /// Default 0 (no accounting) — the byte budget is entry/count bytes only.
    pub fn with_accumulator_cost_per_read(mut self, bytes_per_read: usize) -> Self {
        self.accumulator_bytes_per_read = bytes_per_read;
        self
    }

    /// Number of reads currently buffered.
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Whether the accumulator is empty.
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Approximate heap bytes used by buffered COO entries and counts.
    ///
    /// Does not include `meta`'s own heap allocations (e.g. `String` header
    /// bytes) — callers that need that precision should add it themselves, as
    /// `DeferredDenomBuffer::approx_bytes` does for its `header` field.
    pub fn approx_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<(u64, u32)>()
            + self.fwd_counts.capacity() * std::mem::size_of::<u32>()
            + self.rc_counts.capacity() * std::mem::size_of::<u32>()
    }

    /// `approx_bytes()` plus the projected cost of the classification-pass
    /// accumulator over the reads buffered so far (see
    /// `with_accumulator_cost_per_read`). This, not `approx_bytes()`, is what
    /// `should_flush()` compares against the byte budget.
    pub fn projected_pass_bytes(&self) -> usize {
        self.approx_bytes()
            .saturating_add(self.accumulator_bytes_per_read.saturating_mul(self.meta.len()))
    }

    /// Add a read to the accumulator, flattening its minimizers into flat COO
    /// entries and dropping the per-read `Vec`s immediately.
    ///
    /// # Panics
    /// If this would push the accumulator past `QueryInvertedIndex::MAX_READS`
    /// (2^31 - 1, the packed read-id limit).
    pub fn push(&mut self, meta: M, fwd_mins: Vec<u64>, rc_mins: Vec<u64>) {
        assert!(
            self.meta.len() < QueryInvertedIndex::MAX_READS,
            "QueryAccumulator: read count exceeds MAX_READS ({})",
            QueryInvertedIndex::MAX_READS
        );

        let read_idx = self.meta.len() as u32;
        let fwd_count = fwd_mins.len() as u32;
        let rc_count = rc_mins.len() as u32;

        self.entries.reserve(fwd_mins.len() + rc_mins.len());
        for m in fwd_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, false)));
        }
        for m in rc_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, true)));
        }

        self.fwd_counts.push(fwd_count);
        self.rc_counts.push(rc_count);
        self.meta.push(meta);
    }

    /// Push a whole already-extracted batch (`Vec<(fwd, rc)>`, as produced by
    /// `extract_batch_minimizers`), pairing each read with metadata from `metas`
    /// in order.
    ///
    /// # Panics
    /// If `metas` and `extracted` have different lengths.
    pub fn extend_extracted(
        &mut self,
        metas: impl IntoIterator<Item = M>,
        extracted: Vec<(Vec<u64>, Vec<u64>)>,
    ) {
        let mut metas = metas.into_iter();
        for (fwd, rc) in extracted {
            let meta = metas
                .next()
                .expect("QueryAccumulator::extend_extracted: metas shorter than extracted");
            self.push(meta, fwd, rc);
        }
        assert!(
            metas.next().is_none(),
            "QueryAccumulator::extend_extracted: metas longer than extracted"
        );
    }

    /// Returns true once accumulated data has reached the byte budget, the
    /// optional `max_reads` cap, or the read count is one push away from
    /// `MAX_READS`.
    pub fn should_flush(&self) -> bool {
        self.projected_pass_bytes() >= self.byte_budget
            || self.max_reads.is_some_and(|mr| self.meta.len() >= mr)
            || self.meta.len() >= QueryInvertedIndex::MAX_READS
    }

    /// Drain all buffered data into a sorted `QueryInvertedIndex` and its
    /// per-read metadata, in push order. Preserves allocated capacity for the
    /// next fill cycle.
    ///
    /// Sorts in parallel via rayon — at accumulator scale (hundreds of millions
    /// of entries) a single-threaded sort is a real cost, unlike
    /// `QueryInvertedIndex::build`'s batch-sized sort.
    pub fn drain(&mut self) -> (QueryInvertedIndex, Vec<M>) {
        use rayon::slice::ParallelSliceMut;

        let entry_cap = self.entries.capacity();
        let fwd_cap = self.fwd_counts.capacity();
        let rc_cap = self.rc_counts.capacity();
        let meta_cap = self.meta.capacity();

        let mut entries = std::mem::replace(&mut self.entries, Vec::with_capacity(entry_cap));
        let fwd_counts = std::mem::replace(&mut self.fwd_counts, Vec::with_capacity(fwd_cap));
        let rc_counts = std::mem::replace(&mut self.rc_counts, Vec::with_capacity(rc_cap));
        let meta = std::mem::replace(&mut self.meta, Vec::with_capacity(meta_cap));

        entries.par_sort_unstable_by_key(|&(m, _)| m);

        (
            QueryInvertedIndex::from_sorted_coo(entries, fwd_counts, rc_counts),
            meta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let acc: QueryAccumulator<&str> = QueryAccumulator::with_budget(1024);
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);
        assert!(!acc.should_flush());
    }

    #[test]
    fn test_push_increments_len() {
        let mut acc = QueryAccumulator::with_budget(1_000_000);
        acc.push("r0", vec![100, 200], vec![300]);
        assert_eq!(acc.len(), 1);
        acc.push("r1", vec![], vec![]);
        assert_eq!(acc.len(), 2);
    }

    #[test]
    fn test_should_flush_on_byte_budget() {
        // Small enough that a single 3-entry push exceeds it.
        let mut acc = QueryAccumulator::with_budget(1);
        assert!(!acc.should_flush());
        acc.push("r0", vec![100, 200], vec![300]);
        assert!(acc.should_flush());
    }

    #[test]
    fn test_drain_sorted_and_matches_build() {
        // Same input via QueryAccumulator (incremental) and QueryInvertedIndex::build
        // (one-shot) must produce byte-identical indices.
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![
            (vec![300, 100, 200], vec![250, 150]),
            (vec![100, 400], vec![150]),
            (vec![], vec![]),
            (vec![500], vec![]),
        ];

        let expected = QueryInvertedIndex::build(&queries);

        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        for (i, (fwd, rc)) in queries.into_iter().enumerate() {
            acc.push(format!("read_{i}"), fwd, rc);
        }
        let (actual, metas) = acc.drain();

        assert_eq!(actual.num_entries(), expected.num_entries());
        assert_eq!(actual.num_reads(), expected.num_reads());
        assert_eq!(actual.unique_minimizers(), expected.unique_minimizers());
        for i in 0..expected.num_reads() {
            assert_eq!(actual.fwd_count(i), expected.fwd_count(i));
            assert_eq!(actual.rc_count(i), expected.rc_count(i));
        }
        assert_eq!(metas, vec!["read_0", "read_1", "read_2", "read_3"]);
    }

    #[test]
    fn test_drain_preserves_capacity_and_resets() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("r0", vec![100, 200], vec![300]);
        acc.push("r1", vec![400], vec![500, 600]);

        let (idx, metas) = acc.drain();
        assert_eq!(idx.num_reads(), 2);
        assert_eq!(metas.len(), 2);

        // Accumulator is empty and reusable after drain.
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);

        acc.push("r2", vec![700], vec![]);
        assert_eq!(acc.len(), 1);
        let (idx2, metas2) = acc.drain();
        assert_eq!(idx2.num_reads(), 1);
        assert_eq!(metas2, vec!["r2"]);
    }

    #[test]
    fn test_extend_extracted_matches_sequential_push() {
        let extracted: Vec<(Vec<u64>, Vec<u64>)> =
            vec![(vec![1, 2], vec![3]), (vec![4], vec![5, 6])];
        let metas = vec!["a", "b"];

        let mut via_extend = QueryAccumulator::with_budget(usize::MAX);
        via_extend.extend_extracted(metas.clone(), extracted.clone());

        let mut via_push = QueryAccumulator::with_budget(usize::MAX);
        for (m, (fwd, rc)) in metas.into_iter().zip(extracted) {
            via_push.push(m, fwd, rc);
        }

        let (idx_a, meta_a) = via_extend.drain();
        let (idx_b, meta_b) = via_push.drain();
        assert_eq!(idx_a.num_entries(), idx_b.num_entries());
        assert_eq!(idx_a.unique_minimizers(), idx_b.unique_minimizers());
        assert_eq!(meta_a, meta_b);
    }

    #[test]
    #[should_panic(expected = "metas shorter than extracted")]
    fn test_extend_extracted_rejects_short_metas() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.extend_extracted(
            vec!["only_one"],
            vec![(vec![1], vec![]), (vec![2], vec![])],
        );
    }

    #[test]
    fn test_accumulator_cost_defaults_to_zero() {
        // Without opting in, projected_pass_bytes must equal approx_bytes —
        // existing callers (e.g. the FFI path) see no behavior change.
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("r0", vec![100, 200], vec![300]);
        assert_eq!(acc.projected_pass_bytes(), acc.approx_bytes());
    }

    #[test]
    fn test_should_flush_on_accumulator_cost_even_with_few_entries() {
        // A pass with many reads but few minimizers each (e.g. very short
        // reads against a many-bucket index) can be accumulator-bound rather
        // than entry-bound. should_flush() must catch that even though
        // approx_bytes() alone would stay well under budget.
        let bytes_per_read = 1_300; // ~160-bucket dense accumulator per read
        let budget = 10_000usize;
        let mut acc: QueryAccumulator<&str> =
            QueryAccumulator::with_budget(budget).with_accumulator_cost_per_read(bytes_per_read);

        for _ in 0..7 {
            acc.push("r", vec![1], vec![]);
        }
        // 7 reads * 1300 = 9100 < 10000: not yet.
        assert!(acc.approx_bytes() < budget, "entries alone are tiny");
        assert!(!acc.should_flush());

        acc.push("r", vec![1], vec![]);
        // 8 reads * 1300 = 10400 >= 10000: flush, even though entries are
        // still tiny on their own.
        assert!(acc.should_flush());
    }

    #[test]
    fn test_should_flush_on_max_reads_even_under_byte_budget() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX).with_max_reads(Some(2));
        acc.push("r0", vec![1], vec![]);
        assert!(!acc.should_flush());
        acc.push("r1", vec![2], vec![]);
        assert!(acc.should_flush());
    }

    #[test]
    fn test_zero_minimizer_reads_preserved() {
        // A read shorter than k produces no minimizers; it must still occupy a
        // read slot so downstream indexing by position stays correct.
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("short", vec![], vec![]);
        acc.push("normal", vec![10], vec![20]);

        let (idx, metas) = acc.drain();
        assert_eq!(idx.num_reads(), 2);
        assert_eq!(idx.fwd_count(0), 0);
        assert_eq!(idx.rc_count(0), 0);
        assert_eq!(idx.fwd_count(1), 1);
        assert_eq!(idx.rc_count(1), 1);
        assert_eq!(metas, vec!["short", "normal"]);
    }
}
