//! Offline profiling harness: minimizers vs. syncmers vs. hash-mixed minimizers
//! vs. FracMinHash, for the RY-space sketching used by `rype`.
//!
//! This is measurement code, not a proposed production implementation. It does
//! NOT touch anything under `src/`. See `docs/syncmer-evaluation.md` for the
//! write-up this harness produces the numbers for.
//!
//! Scope, deliberately narrowed for a fair A/B (documented so results aren't
//! over-read):
//!   - Single forward strand only, mirroring the index-build path
//!     (`rype::extract_into`), not the dual-strand query path
//!     (`get_paired_minimizers_into`). Density/size/conservation numbers here
//!     are what would land in the index; real query-side seed counts are ~2x
//!     higher (fwd + rc) but the *ratio* between arms is unaffected.
//!   - "Density" = selected / valid k-mer positions, RAW (no dedup) unless
//!     stated otherwise. "Distinct count" is what actually determines index
//!     size (sorted + deduped over the whole corpus).
//!   - The Parquet byte measurement uses a synthetic bucket_id=0 column, so it
//!     measures the minimizer-column floor for a single-bucket index. A real
//!     multi-bucket index (e.g. n100-w200 with 160 buckets) pays extra for a
//!     non-constant bucket_id column on top of this.
//!
//! Usage:
//!   cargo build --release --example sketch_profile
//!   target/release/examples/sketch_profile --self-test
//!   target/release/examples/sketch_profile genomes --dir <genome_dir> \
//!       --subset 500 --seed 1 --k 64 --w 100 --w 20 --out <out.tsv>
//!   target/release/examples/sketch_profile reads --short <r1.fastq.gz> \
//!       --long <long.fastq.gz> --k 64 --w 100 --w 20 --out <out.tsv> \
//!       [--max-reads 200000]

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use needletail::{parse_fastx_file, FastxReader};

const SALT: u64 = 0x5555555555555555;

// ---------------------------------------------------------------------------
// RNG (splitmix64) - deterministic, dependency-free.
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Order-mixing hash (splitmix64 finalizer). Used only to pick the window
/// minimum / s-mer argmin; the *stored* value stays `kmer ^ salt` in every
/// arm so the delta-encoded Parquet column keeps its structure (finding #1
/// in the plan: the stored value must not become hash noise).
#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

fn hash_str(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn random_ry_seq(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
    (0..len)
        .map(|_| BASES[(rng.next_u64() % 4) as usize])
        .collect()
}

// ---------------------------------------------------------------------------
// Selection arms.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum SyncmerKind {
    Open(usize),
    Closed,
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    MinLex,
    MinHash,
    SyncOpen { s: usize },
    SyncClosed { s: usize },
    FracMin { density: f64 },
}

fn open_target(k: usize, s: usize) -> usize {
    // t = ceil((k - s + 1) / 2), the conservation-optimal offset (Shaw & Yu 2022, Thm 8).
    (k - s + 2) / 2
}

/// Count of valid k-mer positions in `seq` (same reset-on-N accounting as
/// `rype::extract_into`'s `valid_bases_count`). Shared across all arms for a
/// given `k` since it doesn't depend on the selection scheme.
fn valid_kmer_count(seq: &[u8], k: usize) -> u64 {
    let mut run = 0usize;
    let mut count = 0u64;
    for &b in seq {
        if rype::base_to_bit(b) == u64::MAX {
            run = 0;
            continue;
        }
        run += 1;
        if run >= k {
            count += 1;
        }
    }
    count
}

/// Generic windowed-minimum k-mer selector: same monotonic-deque structure
/// and warm-up/dedup semantics as `rype::extract_into`, parameterized by an
/// order-mixing function. With `mix = identity` this MUST reproduce
/// `rype::extract_into` bit-for-bit (asserted in `--self-test`); that's what
/// makes every other arm's comparison to the baseline meaningful.
fn windowed_min_select<F: Fn(u64) -> u64>(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    mix: F,
) -> Vec<u64> {
    let mut out = Vec::new();
    let len = seq.len();
    if len < k {
        return out;
    }
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let mut current_val: u64 = 0;
    let mut valid_bases_count = 0usize;
    let mut last_min: Option<u64> = None;
    // (pos, order_key, stored_value)
    let mut dq: VecDeque<(usize, u64, u64)> = VecDeque::new();

    for (i, &base) in seq.iter().enumerate().take(len) {
        let bit = rype::base_to_bit(base);
        if bit == u64::MAX {
            valid_bases_count = 0;
            dq.clear();
            current_val = 0;
            last_min = None;
            continue;
        }
        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let stored = current_val ^ salt;
            let key = mix(stored);

            while let Some(&(p, _, _)) = dq.front() {
                if p + w <= pos {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk, _)) = dq.back() {
                if kk >= key {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back((pos, key, stored));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, _, min_stored)) = dq.front() {
                    if Some(min_stored) != last_min {
                        out.push(min_stored);
                        last_min = Some(min_stored);
                    }
                }
            }
        }
    }
    out
}

/// Syncmer selector: for each k-mer, find the argmin of its `k-s+1` contained
/// s-mers (via the same monotonic-deque pattern, applied at s-mer
/// granularity) and select the k-mer iff that argmin lands at the target
/// offset (open) or either end (closed).
fn syncmer_select(seq: &[u8], k: usize, s: usize, salt: u64, kind: SyncmerKind) -> Vec<u64> {
    assert!(
        s >= 1 && s < k,
        "s must satisfy 1 <= s < k, got s={s} k={k}"
    );
    let mut out = Vec::new();
    let len = seq.len();
    if len < k {
        return out;
    }
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let s_mask = if s == 64 { u64::MAX } else { (1u64 << s) - 1 };
    let win = k - s + 1;

    let mut current_val: u64 = 0;
    let mut current_val_s: u64 = 0;
    let mut valid_bases_count = 0usize;
    // (s-mer start pos, mixed key)
    let mut dq: VecDeque<(usize, u64)> = VecDeque::new();

    for (i, &base) in seq.iter().enumerate().take(len) {
        let bit = rype::base_to_bit(base);
        if bit == u64::MAX {
            valid_bases_count = 0;
            current_val = 0;
            current_val_s = 0;
            dq.clear();
            continue;
        }
        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_val_s = ((current_val_s << 1) | bit) & s_mask;

        if valid_bases_count >= s {
            let s_pos = i + 1 - s;
            let key = mix64(current_val_s ^ salt);
            while let Some(&(p, _)) = dq.front() {
                if p + win <= s_pos {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = dq.back() {
                if kk >= key {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back((s_pos, key));
        }

        if valid_bases_count >= k {
            let kmer_pos = i + 1 - k;
            if let Some(&(min_pos, _)) = dq.front() {
                let rel = min_pos - kmer_pos;
                let selected = match kind {
                    SyncmerKind::Open(t) => rel == t,
                    SyncmerKind::Closed => rel == 0 || rel == k - s,
                };
                if selected {
                    out.push(current_val ^ salt);
                }
            }
        }
    }
    out
}

/// Context-free control: select each k-mer independently by hashed value vs.
/// a density threshold. No window, no context - the pure "spread guarantee
/// gone, conservation-only" baseline that isolates how much of syncmers'
/// advantage is really about being context-independent vs. just being at a
/// certain density.
fn fracmin_select(seq: &[u8], k: usize, salt: u64, keep_threshold: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let len = seq.len();
    if len < k {
        return out;
    }
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut current_val: u64 = 0;
    let mut valid_bases_count = 0usize;
    for &base in seq.iter().take(len) {
        let bit = rype::base_to_bit(base);
        if bit == u64::MAX {
            valid_bases_count = 0;
            current_val = 0;
            continue;
        }
        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        if valid_bases_count >= k {
            let stored = current_val ^ salt;
            if mix64(stored) < keep_threshold {
                out.push(stored);
            }
        }
    }
    out
}

fn run_arm(seq: &[u8], k: usize, w: usize, salt: u64, arm: Arm) -> Vec<u64> {
    match arm {
        Arm::MinLex => windowed_min_select(seq, k, w, salt, |x| x),
        Arm::MinHash => windowed_min_select(seq, k, w, salt, mix64),
        Arm::SyncOpen { s } => {
            syncmer_select(seq, k, s, salt, SyncmerKind::Open(open_target(k, s)))
        }
        Arm::SyncClosed { s } => syncmer_select(seq, k, s, salt, SyncmerKind::Closed),
        Arm::FracMin { density } => {
            let threshold = (density.clamp(0.0, 1.0) * u64::MAX as f64) as u64;
            fracmin_select(seq, k, salt, threshold)
        }
    }
}

/// Build the arm set for a given (k, w): the baseline, a hash-mixed
/// minimizer control, a small sweep of open/closed syncmers bracketing the
/// window size that would match `min-hash`'s *theoretical* density (the
/// actual `min-lex` density is unknown ahead of time - that's finding #1 -
/// so we sweep rather than solve for a single "matched" s), and two FracMin
/// controls.
fn arm_set_for(k: usize, w: usize) -> Vec<(String, Arm)> {
    let mut arms = vec![
        ("min-lex".to_string(), Arm::MinLex),
        ("min-hash".to_string(), Arm::MinHash),
    ];

    // Same s sweep for open AND closed syncmers, so every density point is
    // directly comparable between the two kinds (open-vs-closed is the
    // question, not just "does either reach the target density").
    let target_open_win = ((w as f64 + 1.0) / 2.0).round() as i64;
    let mut s_values: Vec<usize> = Vec::new();
    for delta in [-1i64, 0, 1, 3, 6, 10] {
        let win = target_open_win + delta;
        if win >= 2 && (win as usize) <= k {
            let s = k as i64 - win + 1;
            if s >= 4 && (s as usize) < k {
                s_values.push(s as usize);
            }
        }
    }
    s_values.sort_unstable();
    s_values.dedup();
    for &s in &s_values {
        arms.push((format!("sync-open-s{s}"), Arm::SyncOpen { s }));
        arms.push((format!("sync-closed-s{s}"), Arm::SyncClosed { s }));
    }

    let d0 = 2.0 / (w as f64 + 1.0);
    arms.push(("fracmin-1x".to_string(), Arm::FracMin { density: d0 }));
    arms.push((
        "fracmin-0.5x".to_string(),
        Arm::FracMin { density: d0 * 0.5 },
    ));

    arms
}

// ---------------------------------------------------------------------------
// RY-aware mutation model.
//
// In RY space only transversions (A<->T, A<->C, G<->T, G<->C) change the
// encoded bit; transitions (A<->G, T<->C) are invisible to every arm here.
// A uniform-substitution model would make every arm look ~2x worse than
// reality and hide the actual RY-space conservation difference between
// schemes. p_transition = kappa / (kappa + 2); kappa=4.0 gives a 2:1
// transition:transversion ratio (a commonly cited empirical value), i.e.
// P(RY-bit flip | substitution) = 1/3.
// ---------------------------------------------------------------------------

const KAPPA: f64 = 4.0;

fn transition_partner(b: u8) -> u8 {
    match b {
        b'A' => b'G',
        b'G' => b'A',
        b'T' => b'C',
        b'C' => b'T',
        other => other,
    }
}

fn transversion_partners(b: u8) -> [u8; 2] {
    match b {
        b'A' | b'G' => [b'T', b'C'],
        b'T' | b'C' => [b'A', b'G'],
        _ => [b'N', b'N'],
    }
}

fn mutate_ry_aware(seq: &[u8], theta: f64, rng: &mut SplitMix64) -> Vec<u8> {
    let p_transition = KAPPA / (KAPPA + 2.0);
    seq.iter()
        .map(|&b| {
            let bu = b.to_ascii_uppercase();
            if !matches!(bu, b'A' | b'C' | b'G' | b'T') {
                return b;
            }
            if rng.next_f64() >= theta {
                return b;
            }
            if rng.next_f64() < p_transition {
                transition_partner(bu)
            } else {
                let tv = transversion_partners(bu);
                if rng.next_u64() % 2 == 0 {
                    tv[0]
                } else {
                    tv[1]
                }
            }
        })
        .collect()
}

fn intersect_count(a: &[u64], b: &[u64]) -> u64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut c = 0u64;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Parquet byte-cost measurement: writes the sorted, deduped minimizer set
// (plus a synthetic all-zero bucket_id column) with production's own
// `ParquetWriteOptions::default()` (Snappy, 100K row groups, no bloom
// filter - the same options `rype index from-config` uses; see
// `src/indices/parquet/options.rs`), and returns the resulting file size.
// This is not a proxy - it's what the shard column would actually cost.
// ---------------------------------------------------------------------------

fn measure_parquet_bytes(sorted_dedup: &[u64], tmp_path: &Path) -> u64 {
    use arrow::array::{UInt32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![
        Field::new("minimizer", DataType::UInt64, false),
        Field::new("bucket_id", DataType::UInt32, false),
    ]));
    let min_col = UInt64Array::from_iter_values(sorted_dedup.iter().copied());
    let bucket_col =
        UInt32Array::from_iter_values(std::iter::repeat(0u32).take(sorted_dedup.len()));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(min_col), Arc::new(bucket_col)],
    )
    .expect("record batch construction");

    let props = rype::ParquetWriteOptions::default().to_writer_properties();
    let file = File::create(tmp_path).unwrap_or_else(|e| panic!("create {tmp_path:?}: {e}"));
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("arrow writer");

    let n = batch.num_rows();
    let rg = 100_000usize;
    let mut off = 0;
    while off < n {
        let len = rg.min(n - off);
        writer
            .write(&batch.slice(off, len))
            .expect("write row group");
        off += len;
    }
    writer.close().expect("close writer");

    std::fs::metadata(tmp_path).map(|m| m.len()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Genome loading.
// ---------------------------------------------------------------------------

fn list_genome_files(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.to_string_lossy();
            name.ends_with(".fasta.gz")
                || name.ends_with(".fa.gz")
                || name.ends_with(".fasta")
                || name.ends_with(".fa")
        })
        .collect();
    files.sort();
    files
}

fn deterministic_subset(files: &[PathBuf], n: usize, seed: u64) -> Vec<PathBuf> {
    let mut idx: Vec<usize> = (0..files.len()).collect();
    let mut rng = SplitMix64::new(seed);
    let take = n.min(idx.len());
    for i in 0..take {
        let j = i + (rng.next_u64() as usize) % (idx.len() - i);
        idx.swap(i, j);
    }
    idx.truncate(take);
    idx.into_iter().map(|i| files[i].clone()).collect()
}

/// Concatenate all contigs in a genome file, separated by a run of `k` `N`s
/// so no spurious k-mer spans a contig boundary (mirrors how a real reset
/// would occur: `N` is a hard separator for `base_to_bit`).
fn read_genome_concat(path: &Path, k: usize) -> Vec<u8> {
    let mut reader = parse_fastx_file(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let sep = vec![b'N'; k];
    let mut seq = Vec::new();
    let mut first = true;
    while let Some(rec) = reader.next() {
        let rec = rec.unwrap_or_else(|e| panic!("record in {path:?}: {e}"));
        if !first {
            seq.extend_from_slice(&sep);
        }
        seq.extend_from_slice(&rec.seq());
        first = false;
    }
    seq
}

// ---------------------------------------------------------------------------
// Self-test.
// ---------------------------------------------------------------------------

fn edge_cases(k: usize, w: usize) -> Vec<Vec<u8>> {
    let mut v = vec![];
    v.push(vec![b'A'; (k + w).saturating_sub(2)]); // shorter than warm-up -> should yield nothing
    v.push(vec![b'A'; k + w + 50]); // homopolymer, long enough
    let mut with_n = random_ry_seq(300.max(k + w + 10), 7);
    let mid = with_n.len() / 2;
    with_n[mid] = b'N';
    v.push(with_n);
    v.push(vec![b'A'; k.saturating_sub(1)]); // shorter than k
    v
}

fn check_close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = (got - want).abs() / want;
    eprintln!("[self-test] {name}: got={got:.5} want={want:.5} rel_err={rel:.3}");
    assert!(
        rel <= tol,
        "{name} density law violated: got {got:.5}, want {want:.5} (rel err {rel:.3} > tol {tol})"
    );
}

fn self_test() {
    let k = 64usize;
    let w = 50usize;
    let salt = SALT;

    // 1. The generic windowed-min selector (mix=identity) must reproduce
    //    rype::extract_into bit-for-bit. This is the load-bearing check: it's
    //    what makes every other arm's comparison to the baseline meaningful.
    let seq = random_ry_seq(500_000, 42);
    let mine = windowed_min_select(&seq, k, w, salt, |x| x);
    let mut ws = rype::MinimizerWorkspace::new();
    rype::extract_into(&seq, k, w, salt, &mut ws);
    assert_eq!(
        mine, ws.buffer,
        "windowed_min_select must match rype::extract_into on random sequence"
    );

    for (i, case) in edge_cases(k, w).into_iter().enumerate() {
        let mine = windowed_min_select(&case, k, w, salt, |x| x);
        let mut ws2 = rype::MinimizerWorkspace::new();
        rype::extract_into(&case, k, w, salt, &mut ws2);
        assert_eq!(
            mine,
            ws2.buffer,
            "edge case #{i} (len={}) mismatch vs rype::extract_into",
            case.len()
        );
    }
    eprintln!(
        "[self-test] windowed_min_select matches rype::extract_into (random + 4 edge cases): OK"
    );

    // 2. Density laws on uniform-random RY sequence, each within tolerance.
    let total_valid = valid_kmer_count(&seq, k) as f64;
    let dens = |sel: usize| sel as f64 / total_valid;

    let mh = windowed_min_select(&seq, k, w, salt, mix64);
    check_close(
        "min-hash density",
        dens(mh.len()),
        2.0 / (w as f64 + 1.0),
        0.15,
    );

    let s = 15usize;
    let so = syncmer_select(&seq, k, s, salt, SyncmerKind::Open(open_target(k, s)));
    check_close(
        "sync-open(s=15) density",
        dens(so.len()),
        1.0 / (k - s + 1) as f64,
        0.15,
    );

    let sc = syncmer_select(&seq, k, s, salt, SyncmerKind::Closed);
    check_close(
        "sync-closed(s=15) density",
        dens(sc.len()),
        2.0 / (k - s + 1) as f64,
        0.15,
    );

    let target = 0.05;
    let fm = fracmin_select(&seq, k, salt, (target * u64::MAX as f64) as u64);
    check_close("fracmin(0.05) density", dens(fm.len()), target, 0.15);

    // min-lex density is deliberately NOT asserted - measuring it is the point
    // (finding #1: the lexicographic order's true density is unknown).
    eprintln!(
        "[self-test] min-lex density (unasserted, informational): {:.5}",
        dens(windowed_min_select(&seq, k, w, salt, |x| x).len())
    );

    eprintln!("[self-test] ALL PASS");
}

// ---------------------------------------------------------------------------
// `genomes` subcommand: density, index-size proxy, conservation under
// RY-aware mutation.
// ---------------------------------------------------------------------------

struct GenomesArgs {
    dir: String,
    subset: usize,
    seed: u64,
    k: usize,
    ws: Vec<usize>,
    out: String,
    conservation_genomes: usize,
}

fn parse_genomes_args(args: &[String]) -> GenomesArgs {
    let mut dir = None;
    let mut subset = 500usize;
    let mut seed = 1u64;
    let mut k = 64usize;
    let mut ws = vec![];
    let mut out = None;
    let mut conservation_genomes = 50usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--subset" => {
                subset = args[i + 1].parse().expect("--subset must be a number");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed must be a number");
                i += 2;
            }
            "--k" => {
                k = args[i + 1].parse().expect("--k must be a number");
                i += 2;
            }
            "--w" => {
                ws.push(args[i + 1].parse().expect("--w must be a number"));
                i += 2;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--conservation-genomes" => {
                conservation_genomes = args[i + 1]
                    .parse()
                    .expect("--conservation-genomes must be a number");
                i += 2;
            }
            other => panic!("unknown arg to `genomes`: {other}"),
        }
    }
    GenomesArgs {
        dir: dir.expect("--dir is required"),
        subset,
        seed,
        k,
        ws: if ws.is_empty() { vec![100] } else { ws },
        out: out.expect("--out is required"),
        conservation_genomes,
    }
}

fn run_genomes(args: &[String]) {
    let a = parse_genomes_args(args);
    let files = list_genome_files(&a.dir);
    assert!(!files.is_empty(), "no genome files found in {}", a.dir);
    let subset_files = deterministic_subset(&files, a.subset, a.seed);
    eprintln!(
        "[genomes] loading {} / {} genome files (seed={})",
        subset_files.len(),
        files.len(),
        a.seed
    );

    let t_load = Instant::now();
    let seqs: Vec<Vec<u8>> = subset_files
        .iter()
        .map(|p| read_genome_concat(p, a.k))
        .collect();
    let total_bases: u64 = seqs.iter().map(|s| s.len() as u64).sum();
    eprintln!(
        "[genomes] loaded {} genomes, {total_bases} bases in {:.1}s",
        seqs.len(),
        t_load.elapsed().as_secs_f64()
    );

    if let Some(parent) = Path::new(&a.out).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = File::create(&a.out).unwrap_or_else(|e| panic!("create {}: {e}", a.out));
    let tmp_parquet = Path::new(&a.out).with_extension("parquet.tmp");

    writeln!(
        f,
        "# sketch_profile genomes: k={} n_genomes={} total_bases={} salt=0x{:016x} kappa={KAPPA}",
        a.k,
        seqs.len(),
        total_bases,
        SALT
    )
    .unwrap();

    writeln!(f, "\n# density_size").unwrap();
    writeln!(
        f,
        "w\targ\ttotal_valid_kmers\ttotal_selected_raw\tdensity\tdistinct_count\tparquet_bytes\tbytes_per_record\tmbase_per_sec"
    )
    .unwrap();

    for &w in &a.ws {
        let total_valid: u64 = seqs.iter().map(|s| valid_kmer_count(s, a.k)).sum();
        for (name, arm) in arm_set_for(a.k, w) {
            let t0 = Instant::now();
            let mut total_selected = 0u64;
            let mut all_vals: Vec<u64> = Vec::new();
            for s in &seqs {
                let sel = run_arm(s, a.k, w, SALT, arm);
                total_selected += sel.len() as u64;
                all_vals.extend(sel);
            }
            let elapsed = t0.elapsed().as_secs_f64();

            all_vals.sort_unstable();
            all_vals.dedup();
            let distinct = all_vals.len() as u64;
            let bytes = measure_parquet_bytes(&all_vals, &tmp_parquet);
            let density = total_selected as f64 / total_valid as f64;
            let bpr = bytes as f64 / distinct.max(1) as f64;
            let mbps = (total_bases as f64 / 1e6) / elapsed.max(1e-9);

            writeln!(f, "{w}\t{name}\t{total_valid}\t{total_selected}\t{density:.6}\t{distinct}\t{bytes}\t{bpr:.3}\t{mbps:.1}").unwrap();
            eprintln!("[genomes] w={w:<4} arm={name:<16} density={density:.5} distinct={distinct:>10} bytes/rec={bpr:.2} ({elapsed:.1}s)");
        }
    }
    f.flush().unwrap();

    writeln!(f, "\n# conservation").unwrap();
    writeln!(
        f,
        "w\targ\ttheta\torig_distinct\tshared_distinct\tconservation\tshared_per_kb"
    )
    .unwrap();

    let cons_n = a.conservation_genomes.min(seqs.len());
    let cons_seqs = &seqs[..cons_n];
    let cons_bases: u64 = cons_seqs.iter().map(|s| s.len() as u64).sum();
    let thetas: [f64; 5] = [0.005, 0.01, 0.02, 0.05, 0.10];
    let n_pairs = cons_n / 2;
    let pair_bases: u64 = cons_seqs[..n_pairs * 2]
        .iter()
        .map(|s| s.len() as u64)
        .sum();
    // Cross-genome specificity: background sharing between *unrelated*
    // genomes (adjacent pairs in the already seed-shuffled subset order, so
    // pairing is effectively random without needing a second RNG draw). This
    // is the failure mode that matters for multi-bucket taxonomic
    // classification (a spurious shared minimizer between two different
    // buckets' genomes pollutes that bucket's score) but is invisible to
    // single-index host filtration, which only cares about same-genome
    // conservation under divergence (the block above). Buffered and written
    // as its own section after all conservation rows, not interleaved.
    let mut specificity_rows: Vec<String> = Vec::new();

    for &w in &a.ws {
        let arms = arm_set_for(a.k, w);
        for (name, arm) in &arms {
            // Original (unmutated) selection per genome, computed once per
            // arm and reused across all thetas (previously recomputed per
            // theta) and for the specificity pairing below.
            let orig_sels: Vec<Vec<u64>> = cons_seqs
                .iter()
                .map(|s| {
                    let mut sel = run_arm(s, a.k, w, SALT, *arm);
                    sel.sort_unstable();
                    sel.dedup();
                    sel
                })
                .collect();

            for theta in thetas {
                let mut rng =
                    SplitMix64::new(a.seed ^ theta.to_bits() ^ (w as u64) ^ hash_str(name));
                let mut shared = 0u64;
                let mut orig_total = 0u64;
                for (gi, s) in cons_seqs.iter().enumerate() {
                    let mutated = mutate_ry_aware(s, theta, &mut rng);
                    let mut mut_sel = run_arm(&mutated, a.k, w, SALT, *arm);
                    mut_sel.sort_unstable();
                    mut_sel.dedup();

                    shared += intersect_count(&orig_sels[gi], &mut_sel);
                    orig_total += orig_sels[gi].len() as u64;
                }
                let conservation = shared as f64 / orig_total.max(1) as f64;
                let shared_per_kb = shared as f64 / (cons_bases as f64 / 1000.0);
                writeln!(f, "{w}\t{name}\t{theta}\t{orig_total}\t{shared}\t{conservation:.5}\t{shared_per_kb:.3}").unwrap();
            }

            let mut spec_shared = 0u64;
            let mut spec_denom = 0u64;
            for p in 0..n_pairs {
                let sa = &orig_sels[2 * p];
                let sb = &orig_sels[2 * p + 1];
                spec_shared += intersect_count(sa, sb);
                spec_denom += sa.len().min(sb.len()) as u64;
            }
            let collision_rate = spec_shared as f64 / spec_denom.max(1) as f64;
            let spec_shared_per_kb = spec_shared as f64 / (pair_bases as f64 / 1000.0);
            specificity_rows.push(format!(
                "{w}\t{name}\t{n_pairs}\t{spec_shared}\t{collision_rate:.6}\t{spec_shared_per_kb:.3}"
            ));

            eprintln!("[genomes] conservation+specificity w={w} arm={name} done");
        }
    }

    writeln!(f, "\n# specificity").unwrap();
    writeln!(
        f,
        "w\targ\tn_pairs\tshared_distinct_total\tbackground_collision_rate\tshared_per_kb"
    )
    .unwrap();
    for row in &specificity_rows {
        writeln!(f, "{row}").unwrap();
    }
    f.flush().unwrap();
    std::fs::remove_file(&tmp_parquet).ok();
    eprintln!("[genomes] wrote {}", a.out);
}

// ---------------------------------------------------------------------------
// `reads` subcommand: zero-seed fraction and seeds/read on real reads.
// ---------------------------------------------------------------------------

struct ReadsArgs {
    short: Option<String>,
    long: Option<String>,
    k: usize,
    ws: Vec<usize>,
    out: String,
    max_reads: usize,
}

fn parse_reads_args(args: &[String]) -> ReadsArgs {
    let mut short = None;
    let mut long = None;
    let mut k = 64usize;
    let mut ws = vec![];
    let mut out = None;
    let mut max_reads = 200_000usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--short" => {
                short = Some(args[i + 1].clone());
                i += 2;
            }
            "--long" => {
                long = Some(args[i + 1].clone());
                i += 2;
            }
            "--k" => {
                k = args[i + 1].parse().expect("--k must be a number");
                i += 2;
            }
            "--w" => {
                ws.push(args[i + 1].parse().expect("--w must be a number"));
                i += 2;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--max-reads" => {
                max_reads = args[i + 1].parse().expect("--max-reads must be a number");
                i += 2;
            }
            other => panic!("unknown arg to `reads`: {other}"),
        }
    }
    assert!(
        short.is_some() || long.is_some(),
        "at least one of --short/--long is required"
    );
    ReadsArgs {
        short,
        long,
        k,
        ws: if ws.is_empty() { vec![100] } else { ws },
        out: out.expect("--out is required"),
        max_reads,
    }
}

fn run_reads(args: &[String]) {
    let a = parse_reads_args(args);
    if let Some(parent) = Path::new(&a.out).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = File::create(&a.out).unwrap_or_else(|e| panic!("create {}: {e}", a.out));
    writeln!(
        f,
        "# sketch_profile reads: k={} salt=0x{:016x} max_reads={}",
        a.k, SALT, a.max_reads
    )
    .unwrap();
    writeln!(
        f,
        "label\tw\targ\tn_reads\tzero_seed_reads\tzero_seed_frac\tmean_seeds_per_read"
    )
    .unwrap();

    for (label, path) in [("short", &a.short), ("long", &a.long)] {
        let Some(path) = path else { continue };
        eprintln!("[reads] {label}: {path}");
        let mut reader: Box<dyn FastxReader> =
            parse_fastx_file(path).unwrap_or_else(|e| panic!("open {path}: {e}"));

        // One pass per read, computing every (w, arm) at once to avoid re-decompressing.
        let arm_sets: Vec<(usize, Vec<(String, Arm)>)> =
            a.ws.iter().map(|&w| (w, arm_set_for(a.k, w))).collect();
        let mut n_reads = 0usize;
        let mut zero: Vec<Vec<u64>> = arm_sets
            .iter()
            .map(|(_, arms)| vec![0u64; arms.len()])
            .collect();
        let mut seed_sum: Vec<Vec<u64>> = arm_sets
            .iter()
            .map(|(_, arms)| vec![0u64; arms.len()])
            .collect();

        while n_reads < a.max_reads {
            let Some(rec) = reader.next() else { break };
            let rec = rec.unwrap_or_else(|e| panic!("record in {path}: {e}"));
            let seq = rec.seq();
            for (wi, (w, arms)) in arm_sets.iter().enumerate() {
                for (ai, (_, arm)) in arms.iter().enumerate() {
                    let mut sel = run_arm(&seq, a.k, *w, SALT, *arm);
                    sel.sort_unstable();
                    sel.dedup();
                    if sel.is_empty() {
                        zero[wi][ai] += 1;
                    }
                    seed_sum[wi][ai] += sel.len() as u64;
                }
            }
            n_reads += 1;
        }

        for (wi, (w, arms)) in arm_sets.iter().enumerate() {
            for (ai, (name, _)) in arms.iter().enumerate() {
                let zf = zero[wi][ai] as f64 / n_reads.max(1) as f64;
                let mean = seed_sum[wi][ai] as f64 / n_reads.max(1) as f64;
                writeln!(
                    f,
                    "{label}\t{w}\t{name}\t{n_reads}\t{}\t{zf:.5}\t{mean:.2}",
                    zero[wi][ai]
                )
                .unwrap();
            }
        }
        eprintln!("[reads] {label}: {n_reads} reads processed");
    }
    f.flush().unwrap();
    eprintln!("[reads] wrote {}", a.out);
}

// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!(
        "usage:\n  sketch_profile --self-test\n  sketch_profile genomes --dir <dir> --subset N --seed S --k K --w W [--w W ...] --out <file> [--conservation-genomes N]\n  sketch_profile reads --short <r1.fastq.gz> --long <long.fastq.gz> --k K --w W [--w W ...] --out <file> [--max-reads N]"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(1);
    }
    match args[0].as_str() {
        "--self-test" => self_test(),
        "genomes" => run_genomes(&args[1..]),
        "reads" => run_reads(&args[1..]),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}
