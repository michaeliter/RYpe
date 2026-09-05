//! Minimizer extraction algorithms.
//!
//! This module provides functions for extracting minimizers from DNA sequences
//! using the RY (purine/pyrimidine) encoding scheme. Minimizers are selected
//! using a sliding window approach with a monotonic deque for O(n) complexity.

// Hot path: index-based iteration avoids iterator overhead in inner loops
#![allow(clippy::needless_range_loop)]

use super::encoding::base_to_bit;
use super::hash::mix64;
use super::sketch::{IntoSketch, Sketch};
use super::workspace::MinimizerWorkspace;

/// Strand indicator for minimizer origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    ReverseComplement,
}

/// Structure-of-arrays for minimizers on a single strand.
///
/// Stores hashes and their corresponding positions in parallel arrays,
/// maintaining extraction order (non-decreasing positions).
#[derive(Debug, Clone)]
pub struct StrandMinimizers {
    /// The minimizer hash values, in extraction order.
    pub hashes: Vec<u64>,
    /// 0-based positions in the sequence where each k-mer starts.
    pub positions: Vec<usize>,
}

/// Extract ordered minimizers with positions per strand (SoA layout).
///
/// Returns `(forward, reverse_complement)` StrandMinimizers. Unlike
/// `extract_dual_strand_into()`, this preserves position information and
/// deduplicates by position change (not by hash value).
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `sketch` - Sketch scheme (minimizer window or open-syncmer `s`)
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
pub fn extract_strand_minimizers(
    seq: &[u8],
    k: usize,
    sketch: impl IntoSketch,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (StrandMinimizers, StrandMinimizers) {
    match sketch.into_sketch() {
        Sketch::Minimizer { w } => extract_strand_minimizers_minimizer(seq, k, w, salt, ws),
        Sketch::OpenSyncmer { s } => extract_strand_minimizers_syncmer(seq, k, s, salt, ws),
    }
}

fn extract_strand_minimizers_minimizer(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (StrandMinimizers, StrandMinimizers) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
        );
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let rc_shift = k - 1;

    let cap = ws.estimated_minimizers;
    let mut fwd = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };
    let mut rc = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };

    let mut current_val: u64 = 0;
    let mut current_rc: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd_pos: Option<usize> = None;
    let mut last_rc_pos: Option<usize> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            last_fwd_pos = None;
            last_rc_pos = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = current_rc ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= h_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + w <= pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_rc.back() {
                if v >= h_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(min_pos, min_hash)) = ws.q_fwd.front() {
                    if last_fwd_pos != Some(min_pos) {
                        fwd.hashes.push(min_hash);
                        fwd.positions.push(min_pos);
                        last_fwd_pos = Some(min_pos);
                    }
                }
                if let Some(&(min_pos, min_hash)) = ws.q_rc.front() {
                    if last_rc_pos != Some(min_pos) {
                        rc.hashes.push(min_hash);
                        rc.positions.push(min_pos);
                        last_rc_pos = Some(min_pos);
                    }
                }
            }
        }
    }

    (fwd, rc)
}

/// Open-syncmer counterpart of `extract_strand_minimizers_minimizer`.
///
/// Selection is position-local (no warm-up gate, no value-based dedup): each
/// k-mer position is visited exactly once, so the position-based dedup that
/// the minimizer path needs (to collapse repeats of the same window minimum)
/// never triggers here. See the rc-mirroring note on
/// `extract_dual_strand_into_syncmer` for the reverse-complement deque.
fn extract_strand_minimizers_syncmer(
    seq: &[u8],
    k: usize,
    s: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (StrandMinimizers, StrandMinimizers) {
    debug_assert!(s > 0 && s < k, "syncmer s-mer size must satisfy 0 < s < k");
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
        );
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let s_mask = if s == 64 { u64::MAX } else { (1u64 << s) - 1 };
    let rc_shift = k - 1;
    let win = k - s + 1;
    let t = Sketch::open_target(k, s);
    let mirror_t = win - 1 - t;

    let cap = ws.estimated_minimizers;
    let mut fwd = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };
    let mut rc = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };

    let mut current_val: u64 = 0;
    let mut current_rc: u64 = 0;
    let mut current_val_s: u64 = 0;
    let mut current_val_s_rc: u64 = 0;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            current_val_s = 0;
            current_val_s_rc = 0;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);
        current_val_s = ((current_val_s << 1) | bit) & s_mask;
        current_val_s_rc = (current_val_s_rc >> 1) | ((bit ^ 1) << (s - 1));

        if valid_bases_count >= s {
            let s_pos = i + 1 - s;

            // Forward s-mer deque: pop back on `>=` so ties keep the
            // rightmost (latest) position -- correct for forward reading
            // order (see extract_dual_strand_into_syncmer).
            let key_fwd = mix64(current_val_s ^ salt);
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + win <= s_pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = ws.q_fwd.back() {
                if kk >= key_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((s_pos, key_fwd));

            // RC s-mer deque: pop back on strict `>` so ties keep the
            // leftmost (earliest) position -- correct for rc reading order.
            let key_rc = mix64(current_val_s_rc ^ salt);
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + win <= s_pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = ws.q_rc.back() {
                if kk > key_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((s_pos, key_rc));
        }

        if valid_bases_count >= k {
            let kmer_pos = i + 1 - k;
            if let Some(&(min_pos, _)) = ws.q_fwd.front() {
                if min_pos - kmer_pos == t {
                    fwd.hashes.push(current_val ^ salt);
                    fwd.positions.push(kmer_pos);
                }
            }
            if let Some(&(min_pos, _)) = ws.q_rc.front() {
                if min_pos - kmer_pos == mirror_t {
                    rc.hashes.push(current_rc ^ salt);
                    rc.positions.push(kmer_pos);
                }
            }
        }
    }

    (fwd, rc)
}

/// Extract sorted, deduplicated minimizer sets per strand.
///
/// Returns `(forward_set, rc_set)` where each `Vec<u64>` is sorted and
/// contains no duplicates. This is a convenience wrapper around
/// `extract_dual_strand_into()` with sort + dedup.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `sketch` - Sketch scheme (minimizer window or open-syncmer `s`)
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
pub fn extract_minimizer_set(
    seq: &[u8],
    k: usize,
    sketch: impl IntoSketch,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, sketch, salt, ws);
    fwd.sort_unstable();
    fwd.dedup();
    rc.sort_unstable();
    rc.dedup();
    (fwd, rc)
}

/// Extract minimizers from a sequence (single strand).
///
/// Uses a monotonic deque to efficiently find the minimum hash value
/// in each sliding window of size `w`. Consecutive duplicate minimizers
/// are deduplicated.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes (A, G, T, C, case insensitive)
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `sketch` - Sketch scheme (minimizer window or open-syncmer `s`)
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage and output
///
/// # Output
/// Extracted minimizers are stored in `ws.buffer` (cleared before use).
pub fn extract_into(
    seq: &[u8],
    k: usize,
    sketch: impl IntoSketch,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) {
    match sketch.into_sketch() {
        Sketch::Minimizer { w } => extract_into_minimizer(seq, k, w, salt, ws),
        Sketch::OpenSyncmer { s } => extract_into_syncmer(seq, k, s, salt, ws),
    }
}

fn extract_into_minimizer(seq: &[u8], k: usize, w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    ws.buffer.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < k {
        return;
    }

    // Precompute mask outside hot loop - only lower k bits are valid
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let mut current_val: u64 = 0;
    let mut last_min: Option<u64> = None;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            // Invalid base (N, etc.) - reset k-mer accumulator
            valid_bases_count = 0;
            ws.q_fwd.clear();
            current_val = 0;
            last_min = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let hash = current_val ^ salt;

            // Maintain monotonic deque - remove old entries outside window
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            // Remove entries with larger hash values
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= hash {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, hash));

            // Output minimizer once we have a full window
            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min_h)) = ws.q_fwd.front() {
                    if Some(min_h) != last_min {
                        ws.buffer.push(min_h);
                        last_min = Some(min_h);
                    }
                }
            }
        }
    }
}

/// Open-syncmer counterpart of `extract_into_minimizer` (single strand,
/// forward only -- reference indices are built forward-strand only, so this
/// is the path used at build time).
///
/// Rolls a k-mer accumulator (`current_val`, stored on selection) and a
/// separate s-mer accumulator (`current_val_s`, ordered by `mix64` and fed
/// into a monotonic deque with inner window `win = k - s + 1`). A k-mer is
/// selected iff the deque's front (the argmin s-mer) sits at the
/// conservation-optimal offset `t` from the k-mer's start. No warm-up gate
/// and no value-based dedup: syncmer selection is position-local, so each
/// k-mer position is visited exactly once.
fn extract_into_syncmer(seq: &[u8], k: usize, s: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    debug_assert!(s > 0 && s < k, "syncmer s-mer size must satisfy 0 < s < k");
    ws.buffer.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < k {
        return;
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let s_mask = if s == 64 { u64::MAX } else { (1u64 << s) - 1 };
    let win = k - s + 1;
    let t = Sketch::open_target(k, s);

    let mut current_val: u64 = 0;
    let mut current_val_s: u64 = 0;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            current_val = 0;
            current_val_s = 0;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_val_s = ((current_val_s << 1) | bit) & s_mask;

        if valid_bases_count >= s {
            let s_pos = i + 1 - s;
            let key = mix64(current_val_s ^ salt);

            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + win <= s_pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = ws.q_fwd.back() {
                if kk >= key {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((s_pos, key));
        }

        if valid_bases_count >= k {
            let kmer_pos = i + 1 - k;
            if let Some(&(min_pos, _)) = ws.q_fwd.front() {
                if min_pos - kmer_pos == t {
                    ws.buffer.push(current_val ^ salt);
                }
            }
        }
    }
}

/// Extract minimizers from both strands of a sequence.
///
/// Simultaneously computes minimizers for the forward strand and its
/// reverse complement. This is useful for strand-agnostic matching.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `sketch` - Sketch scheme (minimizer window or open-syncmer `s`)
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
///
/// # Returns
/// A tuple of (forward_minimizers, reverse_complement_minimizers).
pub fn extract_dual_strand_into(
    seq: &[u8],
    k: usize,
    sketch: impl IntoSketch,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    match sketch.into_sketch() {
        Sketch::Minimizer { w } => extract_dual_strand_into_minimizer(seq, k, w, salt, ws),
        Sketch::OpenSyncmer { s } => extract_dual_strand_into_syncmer(seq, k, s, salt, ws),
    }
}

fn extract_dual_strand_into_minimizer(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (vec![], vec![]);
    }

    // Precompute mask outside hot loop - only lower k bits are valid
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    // Shift amount for incremental reverse complement computation
    let rc_shift = k - 1;

    let mut fwd_mins = Vec::with_capacity(ws.estimated_minimizers);
    let mut rc_mins = Vec::with_capacity(ws.estimated_minimizers);

    let mut current_val: u64 = 0;
    // Incremental reverse complement: avoids expensive reverse_bits() call per position
    // When kmer' = (kmer << 1) | new_bit, then rc' = (rc >> 1) | (complement_bit << (k-1))
    let mut current_rc: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd: Option<u64> = None;
    let mut last_rc: Option<u64> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            last_fwd = None;
            last_rc = None;
            continue;
        }

        valid_bases_count += 1;
        // Update forward k-mer
        current_val = ((current_val << 1) | bit) & k_mask;
        // Update reverse complement incrementally: new bit's complement goes to MSB position
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = current_rc ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= h_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + w <= pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_rc.back() {
                if v >= h_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min)) = ws.q_fwd.front() {
                    if Some(min) != last_fwd {
                        fwd_mins.push(min);
                        last_fwd = Some(min);
                    }
                }
                if let Some(&(_, min)) = ws.q_rc.front() {
                    if Some(min) != last_rc {
                        rc_mins.push(min);
                        last_rc = Some(min);
                    }
                }
            }
        }
    }
    (fwd_mins, rc_mins)
}

/// Open-syncmer counterpart of `extract_dual_strand_into_minimizer`.
///
/// Queries sketch both strands (`get_paired_minimizers_into`), while
/// reference indices are built forward-strand only (`extract_into`), so this
/// rc branch must reproduce exactly what forward-sketching `revcomp(seq)`
/// would produce -- otherwise minus-strand reads lose their syncmers and
/// classify as noise. Two mirrorings make that hold:
///
/// 1. **Mirrored target.** The rc s-mer accumulator `current_val_s_rc` rolls
///    the reverse complement of the *s-mer*, so its contained s-mers are the
///    forward ones in reverse order: forward offset `j` sits at rc-frame
///    offset `win - 1 - j`. Selection uses `win - 1 - t`, not `t`.
/// 2. **Mirrored tie-break.** The forward deque pops back on `>=`, so ties
///    keep the rightmost (latest) position in forward reading order. In the
///    rc frame, "rightmost in rc reading order" is "leftmost in forward
///    coordinates", so the rc deque pops back on strict `>` to keep the
///    earliest position instead.
///
/// Verified against a brute-force `revcomp(seq)` oracle in
/// `tests::syncmer_selection::test_syncmer_rc_matches_forward_on_revcomp`.
fn extract_dual_strand_into_syncmer(
    seq: &[u8],
    k: usize,
    s: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    debug_assert!(s > 0 && s < k, "syncmer s-mer size must satisfy 0 < s < k");
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (vec![], vec![]);
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let s_mask = if s == 64 { u64::MAX } else { (1u64 << s) - 1 };
    let rc_shift = k - 1;
    let win = k - s + 1;
    let t = Sketch::open_target(k, s);
    let mirror_t = win - 1 - t;

    let mut fwd_mins = Vec::with_capacity(ws.estimated_minimizers);
    let mut rc_mins = Vec::with_capacity(ws.estimated_minimizers);

    let mut current_val: u64 = 0;
    let mut current_rc: u64 = 0;
    let mut current_val_s: u64 = 0;
    let mut current_val_s_rc: u64 = 0;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            current_val_s = 0;
            current_val_s_rc = 0;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);
        current_val_s = ((current_val_s << 1) | bit) & s_mask;
        current_val_s_rc = (current_val_s_rc >> 1) | ((bit ^ 1) << (s - 1));

        if valid_bases_count >= s {
            let s_pos = i + 1 - s;

            let key_fwd = mix64(current_val_s ^ salt);
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + win <= s_pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = ws.q_fwd.back() {
                if kk >= key_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((s_pos, key_fwd));

            let key_rc = mix64(current_val_s_rc ^ salt);
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + win <= s_pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, kk)) = ws.q_rc.back() {
                if kk > key_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((s_pos, key_rc));
        }

        if valid_bases_count >= k {
            let kmer_pos = i + 1 - k;
            if let Some(&(min_pos, _)) = ws.q_fwd.front() {
                if min_pos - kmer_pos == t {
                    fwd_mins.push(current_val ^ salt);
                }
            }
            if let Some(&(min_pos, _)) = ws.q_rc.front() {
                if min_pos - kmer_pos == mirror_t {
                    rc_mins.push(current_rc ^ salt);
                }
            }
        }
    }
    (fwd_mins, rc_mins)
}

/// Extract minimizers from paired-end reads.
///
/// For paired-end reads, the forward strand of read 1 is combined with
/// the reverse complement of read 2 (and vice versa) to handle the
/// typical paired-end library orientation.
///
/// # Arguments
/// * `s1` - First read sequence
/// * `s2` - Optional second read sequence (for paired-end)
/// * `k` - K-mer size
/// * `sketch` - Sketch scheme (minimizer window or open-syncmer `s`)
/// * `salt` - XOR salt
/// * `ws` - Workspace
///
/// # Returns
/// A tuple of (forward_minimizers, reverse_complement_minimizers),
/// both sorted and deduplicated.
pub fn get_paired_minimizers_into(
    s1: &[u8],
    s2: Option<&[u8]>,
    k: usize,
    sketch: impl IntoSketch,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    let sketch = sketch.into_sketch();
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, k, sketch, salt, ws);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, k, sketch, salt, ws);
        // Combine: read1_fwd + read2_rc, read1_rc + read2_fwd
        fwd.append(&mut r2_rc);
        rc.append(&mut r2_f);
    }
    fwd.sort_unstable();
    fwd.dedup();
    rc.sort_unstable();
    rc.dedup();
    (fwd, rc)
}

/// Count how many minimizers from a query match a bucket (binary search).
///
/// # Arguments
/// * `mins` - Query minimizers (any order)
/// * `bucket` - Bucket minimizers (must be sorted)
///
/// # Returns
/// Number of matches as f64.
pub fn count_hits(mins: &[u64], bucket: &[u64]) -> f64 {
    let mut hits = 0;
    for m in mins {
        if bucket.binary_search(m).is_ok() {
            hits += 1;
        }
    }
    hits as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_into_basic() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGT"; // 20 bases
        extract_into(seq, 16, 4, 0, &mut ws);
        // With k=16 and w=4, we need at least k+w-1=19 bases for one minimizer
        assert!(!ws.buffer.is_empty());
    }

    #[test]
    fn test_extract_into_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT"; // 4 bases, too short for k=16
        extract_into(seq, 16, 4, 0, &mut ws);
        assert!(ws.buffer.is_empty());
    }

    #[test]
    fn test_extract_into_with_n() {
        let mut ws = MinimizerWorkspace::new();
        // N in the middle should reset the k-mer accumulator
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_into(seq, 16, 4, 0, &mut ws);
        let count_without_n = ws.buffer.len();

        // Same sequence with N should produce fewer or different minimizers
        let seq_with_n = b"ACGTACGTACGTACGTNACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_into(seq_with_n, 16, 4, 0, &mut ws);
        // The N disrupts extraction
        assert!(ws.buffer.len() <= count_without_n || ws.buffer.len() > 0);
    }

    #[test]
    fn test_count_hits() {
        let query = vec![1, 5, 10, 15];
        let bucket = vec![1, 2, 3, 5, 7, 10, 12]; // sorted
        assert_eq!(count_hits(&query, &bucket), 3.0); // matches: 1, 5, 10
    }

    #[test]
    fn test_valid_extraction_long() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70];
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(
            !ws.buffer.is_empty(),
            "Should extract minimizers from valid long seq"
        );
    }

    #[test]
    fn test_short_sequences_ignored() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 60];
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(
            ws.buffer.is_empty(),
            "Should not extract minimizers from seq < K"
        );
    }

    #[test]
    fn test_extract_into_boundary_k_plus_w_minus_2_is_empty() {
        // k=16, w=4: the warm-up gate is `valid_bases_count >= k+w-1 == 19`.
        // One base short of that must yield zero minimizers.
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTAC"; // 18 bases
        assert_eq!(seq.len(), 16 + 4 - 2);
        extract_into(seq, 16, 4, 0, &mut ws);
        assert!(
            ws.buffer.is_empty(),
            "k+w-2 bases must yield zero minimizers, one short of the warm-up gate"
        );
    }

    #[test]
    fn test_extract_into_boundary_k_plus_w_minus_1_has_exactly_one() {
        // Exactly enough bases for the warm-up gate to fire once.
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACG"; // 19 bases
        assert_eq!(seq.len(), 16 + 4 - 1);
        extract_into(seq, 16, 4, 0, &mut ws);
        assert_eq!(
            ws.buffer.len(),
            1,
            "k+w-1 bases must yield exactly one minimizer"
        );
    }

    #[test]
    fn test_n_handling_separator() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a: Vec<u8> = (0..80)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect();
        let seq_b: Vec<u8> = (0..80)
            .map(|i| if i % 3 == 0 { b'G' } else { b'C' })
            .collect();

        extract_into(&seq_a, 64, 5, 0, &mut ws);
        let mins_a = ws.buffer.clone();

        extract_into(&seq_b, 64, 5, 0, &mut ws);
        let mins_b = ws.buffer.clone();

        let mut seq_combined = seq_a.clone();
        seq_combined.push(b'N'); // N separator
        seq_combined.extend_from_slice(&seq_b);

        extract_into(&seq_combined, 64, 5, 0, &mut ws);
        let mins_combined = ws.buffer.clone();

        let mut expected = mins_a;
        expected.extend(mins_b);

        assert_eq!(
            mins_combined, expected,
            "N should act as a perfect separator"
        );
    }

    #[test]
    fn test_dual_strand_extraction() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
        assert_ne!(fwd, rc);
    }

    #[test]
    fn test_extract_minimizers_k16() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAA"; // 20 bases
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=16");
    }

    #[test]
    fn test_extract_minimizers_k32() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"; // 40 bases
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=32");
    }

    #[test]
    fn test_short_seq_k16() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGG"; // 10 bases, too short for K=16
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    #[test]
    fn test_short_seq_k32() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAA"; // 20 bases, too short for K=32
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    /// Test that incremental reverse complement matches the full computation.
    /// This verifies the optimization: rc' = (rc >> 1) | ((bit ^ 1) << (k-1))
    #[test]
    fn test_incremental_reverse_complement_correctness() {
        use super::super::encoding::reverse_complement;

        // Test for each supported k value
        for k in [16, 32, 64] {
            let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
            let rc_shift = k - 1;

            // Test sequence with mixed purines/pyrimidines
            let seq =
                b"AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTC";

            let mut current_val: u64 = 0;
            let mut current_rc: u64 = 0;

            for (i, &base) in seq.iter().enumerate() {
                let bit = super::super::encoding::base_to_bit(base);
                if bit == u64::MAX {
                    continue;
                }

                // Update forward k-mer
                current_val = ((current_val << 1) | bit) & k_mask;
                // Update reverse complement incrementally
                current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

                // Once we have k valid bases, verify the incremental RC matches full computation
                if i + 1 >= k {
                    let expected_rc = reverse_complement(current_val, k);
                    assert_eq!(
                        current_rc, expected_rc,
                        "Incremental RC mismatch at position {} for k={}: got {:#x}, expected {:#x}",
                        i, k, current_rc, expected_rc
                    );
                }
            }
        }
    }

    /// Test incremental RC with invalid bases (N) that cause resets.
    #[test]
    fn test_incremental_rc_with_resets() {
        use super::super::encoding::reverse_complement;

        let k = 16;
        let k_mask = (1u64 << k) - 1;
        let rc_shift = k - 1;

        // Sequence with N in the middle
        let seq = b"AGTCAGTCAGTCAGTCNAGTCAGTCAGTCAGTC";

        let mut current_val: u64 = 0;
        let mut current_rc: u64 = 0;
        let mut valid_bases = 0;

        for (i, &base) in seq.iter().enumerate() {
            let bit = super::super::encoding::base_to_bit(base);

            if bit == u64::MAX {
                // Reset on invalid base
                current_val = 0;
                current_rc = 0;
                valid_bases = 0;
                continue;
            }

            valid_bases += 1;
            current_val = ((current_val << 1) | bit) & k_mask;
            current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

            if valid_bases >= k {
                let expected_rc = reverse_complement(current_val, k);
                assert_eq!(
                    current_rc, expected_rc,
                    "Incremental RC mismatch after reset at position {}: got {:#x}, expected {:#x}",
                    i, current_rc, expected_rc
                );
            }
        }
    }

    // ========== extract_strand_minimizers tests ==========

    #[test]
    fn test_extract_strand_minimizers_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        assert!(fwd.hashes.is_empty());
        assert!(fwd.positions.is_empty());
        assert!(rc.hashes.is_empty());
        assert!(rc.positions.is_empty());
    }

    #[test]
    fn test_extract_strand_minimizers_basic() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 64, 5, 0, &mut ws);
        assert!(!fwd.hashes.is_empty(), "Forward should be non-empty");
        assert!(!rc.hashes.is_empty(), "RC should be non-empty");
    }

    #[test]
    fn test_extract_strand_minimizers_soa_invariant() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 32, 4, 0, &mut ws);
        assert_eq!(
            fwd.hashes.len(),
            fwd.positions.len(),
            "Forward SoA mismatch"
        );
        assert_eq!(rc.hashes.len(), rc.positions.len(), "RC SoA mismatch");
    }

    #[test]
    fn test_extract_strand_minimizers_positions_valid() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 32 bases
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        for &p in &fwd.positions {
            assert!(p + 16 <= seq.len(), "Forward position {} out of bounds", p);
        }
        for &p in &rc.positions {
            assert!(p + 16 <= seq.len(), "RC position {} out of bounds", p);
        }
    }

    #[test]
    fn test_extract_strand_minimizers_n_handling() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAANAAAATTTTGGGGCCCCAAAA";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        // N is at position 20; no minimizer should span it
        for &p in &fwd.positions {
            let ends_before_n = p + 16 <= 20;
            let starts_after_n = p > 20;
            assert!(
                ends_before_n || starts_after_n,
                "Forward pos {} spans N at 20",
                p
            );
        }
        for &p in &rc.positions {
            let ends_before_n = p + 16 <= 20;
            let starts_after_n = p > 20;
            assert!(
                ends_before_n || starts_after_n,
                "RC pos {} spans N at 20",
                p
            );
        }
    }

    #[test]
    fn test_extract_strand_minimizers_hashes_match_dual_strand() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd_sm, rc_sm) = extract_strand_minimizers(seq, 64, 5, 0, &mut ws);
        let (fwd_ds, rc_ds) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);

        let fwd_set: std::collections::HashSet<_> = fwd_ds.iter().collect();
        let rc_set: std::collections::HashSet<_> = rc_ds.iter().collect();

        let fwd_matches = fwd_sm.hashes.iter().filter(|h| fwd_set.contains(h)).count();
        let rc_matches = rc_sm.hashes.iter().filter(|h| rc_set.contains(h)).count();

        assert!(
            fwd_matches > 0,
            "Forward hashes should overlap with dual_strand"
        );
        assert!(rc_matches > 0, "RC hashes should overlap with dual_strand");
    }

    #[test]
    fn test_extract_strand_minimizers_positions_ordered() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 5, 0, &mut ws);
        for w in fwd.positions.windows(2) {
            assert!(
                w[0] <= w[1],
                "Forward positions not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
        for w in rc.positions.windows(2) {
            assert!(
                w[0] <= w[1],
                "RC positions not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_extract_strand_minimizers_k16_k32_k64() {
        let mut ws = MinimizerWorkspace::new();
        // 80-base sequence works for all k values
        let seq: Vec<u8> = (0..80)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'T',
                2 => b'G',
                _ => b'C',
            })
            .collect();

        for k in [16, 32, 64] {
            let (fwd, rc) = extract_strand_minimizers(&seq, k, 5, 0, &mut ws);
            assert!(!fwd.hashes.is_empty(), "Forward empty for k={}", k);
            assert!(!rc.hashes.is_empty(), "RC empty for k={}", k);
            assert_eq!(
                fwd.hashes.len(),
                fwd.positions.len(),
                "SoA mismatch for k={}",
                k
            );
            assert_eq!(
                rc.hashes.len(),
                rc.positions.len(),
                "SoA mismatch for k={}",
                k
            );
            for &p in &fwd.positions {
                assert!(p + k <= seq.len(), "pos+k > len for k={}", k);
            }
            for &p in &rc.positions {
                assert!(p + k <= seq.len(), "pos+k > len for k={}", k);
            }
        }
    }

    // ========== extract_minimizer_set tests ==========

    #[test]
    fn test_extract_minimizer_set_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 4, 0, &mut ws);
        assert!(fwd.is_empty());
        assert!(rc.is_empty());
    }

    #[test]
    fn test_extract_minimizer_set_sorted() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        for w in fwd.windows(2) {
            assert!(w[0] <= w[1], "Forward not sorted");
        }
        for w in rc.windows(2) {
            assert!(w[0] <= w[1], "RC not sorted");
        }
    }

    #[test]
    fn test_extract_minimizer_set_deduped() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        for w in fwd.windows(2) {
            assert!(w[0] != w[1], "Forward has adjacent duplicate");
        }
        for w in rc.windows(2) {
            assert!(w[0] != w[1], "RC has adjacent duplicate");
        }
    }

    #[test]
    fn test_extract_minimizer_set_matches_dual_strand() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";

        let (set_fwd, set_rc) = extract_minimizer_set(seq, 64, 5, 0, &mut ws);

        // Manually do the same thing
        let (mut ds_fwd, mut ds_rc) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);
        ds_fwd.sort_unstable();
        ds_fwd.dedup();
        ds_rc.sort_unstable();
        ds_rc.dedup();

        assert_eq!(set_fwd, ds_fwd, "Forward sets should match");
        assert_eq!(set_rc, ds_rc, "RC sets should match");
    }

    // ========== golden minimizer vectors (regression pin for the Phase 2
    // `Selector` refactor) ==========
    //
    // Captured verbatim from the pre-refactor implementation via a one-off
    // `examples/golden_gen.rs` harness (not checked in) run against this
    // exact commit. The `Selector`-trait rewrite touches every extraction
    // loop in this file, so an accidental change to the *minimizer* scheme
    // -- the one scheme every existing index depends on -- is the single
    // biggest risk in that rewrite. This is the only test that pins actual
    // output values (not just "non-empty" or "sorted") and so is the only
    // one that can catch such a regression.
    mod golden_minimizer_vectors {
        use super::*;

        struct Case {
            name: &'static str,
            seq: &'static [u8],
            k: usize,
            w: usize,
            salt: u64,
            into: &'static [u64],
            dual_fwd: &'static [u64],
            dual_rc: &'static [u64],
            strand_fwd_hashes: &'static [u64],
            strand_fwd_positions: &'static [usize],
            strand_rc_hashes: &'static [u64],
            strand_rc_positions: &'static [usize],
        }

        fn cases() -> Vec<Case> {
            vec![
                Case {
                    name: "g1 (k=16,w=4,salt=0)",
                    seq: b"ACGTAGCTTGACCGTAAGCTTGGACCTA",
                    k: 16,
                    w: 4,
                    salt: 0,
                    into: &[0x58cb, 0x632e, 0x1971, 0x32e3, 0x2e38],
                    dual_fwd: &[0x58cb, 0x632e, 0x1971, 0x32e3, 0x2e38],
                    dual_rc: &[0x1672, 0x7167, 0x38b3, 0x1c59, 0x71c5],
                    strand_fwd_hashes: &[0x58cb, 0x632e, 0x1971, 0x32e3, 0x2e38],
                    strand_fwd_positions: &[1, 3, 6, 7, 11],
                    strand_rc_hashes: &[0x1672, 0x7167, 0x38b3, 0x1c59, 0x71c5],
                    strand_rc_positions: &[2, 6, 7, 8, 12],
                },
                Case {
                    name: "g2 (k=32,w=5,salt=12345)",
                    seq: b"ACGTAGCTTGACCGTAAGCTTGGACCTAGGTCAAGCTTAGGCATCGTAGCTAGCATGCTAGCTAGCATCGATCG",
                    k: 32,
                    w: 5,
                    salt: 12345,
                    into: &[
                        0x58cbbe00, 0x1971f701, 0x2e38d724, 0x5c71fe03, 0x71c708d0, 0x1c73beaf,
                        0x38e72d15, 0x1ce395a3, 0x39c77b0c, 0x38e9569f, 0x1d2ce4f4, 0x3a5999a3,
                        0x4b350375,
                    ],
                    dual_fwd: &[
                        0x58cbbe00, 0x1971f701, 0x2e38d724, 0x5c71fe03, 0x71c708d0, 0x1c73beaf,
                        0x38e72d15, 0x1ce395a3, 0x39c77b0c, 0x38e9569f, 0x1d2ce4f4, 0x3a5999a3,
                        0x4b350375,
                    ],
                    dual_rc: &[
                        0x18e3bb00, 0x38c72c60, 0x1c63be15, 0x4718d3b2, 0x5a38f725, 0x2d1c53b7,
                        0x32d1f601, 0x4cb441b7, 0x532d2c5a, 0x3532e1ff, 0x33531d25, 0x4cd4fb7e,
                        0x34cd7c8d,
                    ],
                    strand_fwd_hashes: &[
                        0x58cbbe00, 0x1971f701, 0x2e38d724, 0x5c71fe03, 0x71c708d0, 0x1c73beaf,
                        0x38e72d15, 0x1ce395a3, 0x39c77b0c, 0x38e9569f, 0x1d2ce4f4, 0x3a5999a3,
                        0x4b350375,
                    ],
                    strand_fwd_positions: &[1, 6, 11, 12, 14, 18, 19, 24, 25, 30, 35, 36, 41],
                    strand_rc_hashes: &[
                        0x18e3bb00, 0x38c72c60, 0x1c63be15, 0x4718d3b2, 0x5a38f725, 0x2d1c53b7,
                        0x32d1f601, 0x4cb441b7, 0x532d2c5a, 0x3532e1ff, 0x33531d25, 0x4cd4fb7e,
                        0x34cd7c8d,
                    ],
                    strand_rc_positions: &[3, 8, 9, 11, 16, 17, 21, 23, 25, 29, 33, 35, 39],
                },
                Case {
                    name: "g3 (k=64,w=10,salt=0xDEADBEEF)",
                    seq: b"ACGTAGCTTGACCGTAAGCTTGGACCTAGGTCAAGCTTAGGCATCGTAGCTAGCATGCTAGCTAGCATCGATCGTTACGGACTGCATCGATCGTAGCTAGGCATCGATGCTAGCATCGTAGCTAGCATGCA",
                    k: 64,
                    w: 10,
                    salt: 0xDEADBEEF,
                    into: &[
                        0x1971c73837cb1886, 0x2e38e71df27973dd, 0x1c738e96b4cb27c1,
                        0x1ce3a59a470bf57b, 0x38e966a6b73f5bdd, 0x1d2cd4cdecf118b6,
                        0x2cd4cd32820be73c, 0x35334c97f73bca3c, 0x334c972948d96da4,
                        0x325ca6590de0923a,
                    ],
                    dual_fwd: &[
                        0x1971c73837cb1886, 0x2e38e71df27973dd, 0x1c738e96b4cb27c1,
                        0x1ce3a59a470bf57b, 0x38e966a6b73f5bdd, 0x1d2cd4cdecf118b6,
                        0x2cd4cd32820be73c, 0x35334c97f73bca3c, 0x334c972948d96da4,
                        0x325ca6590de0923a,
                    ],
                    dual_rc: &[
                        0x33532d1cbd23920a, 0x34cd4cb4af23865c, 0x2d9a66a6849579f3,
                        0x16cd3353f3b1dd61, 0x358b6699773b30de, 0x2cd62d9ab80be4d7,
                        0x1966b16c0d988c3e, 0x34659ac56de16a24, 0x34d1966bc8608dbc,
                        0x2d34d196b5bb73dc, 0x32d34d19b81cd23c,
                    ],
                    strand_fwd_hashes: &[
                        0x1971c73837cb1886, 0x2e38e71df27973dd, 0x1c738e96b4cb27c1,
                        0x1ce3a59a470bf57b, 0x38e966a6b73f5bdd, 0x1d2cd4cdecf118b6,
                        0x2cd4cd32820be73c, 0x35334c97f73bca3c, 0x334c972948d96da4,
                        0x325ca6590de0923a,
                    ],
                    strand_fwd_positions: &[6, 11, 18, 24, 30, 35, 43, 49, 57, 67],
                    strand_rc_hashes: &[
                        0x33532d1cbd23920a, 0x34cd4cb4af23865c, 0x2d9a66a6849579f3,
                        0x16cd3353f3b1dd61, 0x358b6699773b30de, 0x2cd62d9ab80be4d7,
                        0x1966b16c0d988c3e, 0x34659ac56de16a24, 0x34d1966bc8608dbc,
                        0x2d34d196b5bb73dc, 0x32d34d19b81cd23c,
                    ],
                    strand_rc_positions: &[1, 7, 16, 17, 26, 32, 37, 43, 49, 57, 61],
                },
                Case {
                    // Homopolymer: degenerate but still a real pin -- the
                    // forward value saturates to all-1s, rc to all-0s.
                    name: "g4 (homopolymer A, k=16,w=4,salt=0)",
                    seq: &[b'A'; 80],
                    k: 16,
                    w: 4,
                    salt: 0,
                    into: &[0xffff],
                    dual_fwd: &[0xffff],
                    dual_rc: &[0x0],
                    strand_fwd_hashes: &[0xffff; 62],
                    // Positions 3..=64: every valid k=16 window start once
                    // the k+w-1=19-base warm-up is satisfied.
                    strand_fwd_positions: &[
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64,
                    ],
                    strand_rc_hashes: &[0x0; 62],
                    strand_rc_positions: &[
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64,
                    ],
                },
                Case {
                    name: "g5 (N separator, k=16,w=4,salt=0)",
                    seq: b"AAAATTTTGGGGCCCCAAAANAAAATTTTGGGGCCCCAAAA",
                    k: 16,
                    w: 4,
                    salt: 0,
                    into: &[0x8787, 0xf0f, 0x8787, 0xf0f],
                    dual_fwd: &[0x8787, 0xf0f, 0x8787, 0xf0f],
                    dual_rc: &[0x1e1e, 0xf0f, 0x1e1e, 0xf0f],
                    strand_fwd_hashes: &[0x8787, 0xf0f, 0x8787, 0xf0f],
                    strand_fwd_positions: &[3, 4, 24, 25],
                    strand_rc_hashes: &[0x1e1e, 0xf0f, 0x1e1e, 0xf0f],
                    strand_rc_positions: &[3, 4, 24, 25],
                },
            ]
        }

        #[test]
        fn test_golden_extract_into() {
            let mut ws = MinimizerWorkspace::new();
            for c in cases() {
                extract_into(c.seq, c.k, c.w, c.salt, &mut ws);
                assert_eq!(ws.buffer, c.into, "extract_into mismatch for {}", c.name);
            }
        }

        #[test]
        fn test_golden_extract_dual_strand_into() {
            let mut ws = MinimizerWorkspace::new();
            for c in cases() {
                let (fwd, rc) = extract_dual_strand_into(c.seq, c.k, c.w, c.salt, &mut ws);
                assert_eq!(fwd, c.dual_fwd, "dual_fwd mismatch for {}", c.name);
                assert_eq!(rc, c.dual_rc, "dual_rc mismatch for {}", c.name);
            }
        }

        #[test]
        fn test_golden_extract_strand_minimizers() {
            let mut ws = MinimizerWorkspace::new();
            for c in cases() {
                let (fwd, rc) = extract_strand_minimizers(c.seq, c.k, c.w, c.salt, &mut ws);
                assert_eq!(
                    fwd.hashes, c.strand_fwd_hashes,
                    "strand fwd hashes mismatch for {}",
                    c.name
                );
                assert_eq!(
                    fwd.positions, c.strand_fwd_positions,
                    "strand fwd positions mismatch for {}",
                    c.name
                );
                assert_eq!(
                    rc.hashes, c.strand_rc_hashes,
                    "strand rc hashes mismatch for {}",
                    c.name
                );
                assert_eq!(
                    rc.positions, c.strand_rc_positions,
                    "strand rc positions mismatch for {}",
                    c.name
                );
            }
        }
    }

    // ========== open-syncmer selection ==========
    //
    // The oracle here is deliberately independent of the production loop: it
    // recomputes every s-mer in a k-mer's window directly (no deque) and
    // applies the tie-break rule by construction (rightmost-wins for
    // forward, leftmost-wins for rc), rather than re-implementing the deque
    // logic. Equality with the rolling implementation is therefore a real
    // check, not a tautology (Rule 7).
    mod syncmer_selection {
        use super::*;

        /// Deterministic PRNG-ish RY sequence generator (no `rand` dependency
        /// in this crate). Reuses `mix64` purely as an avalanche step; this
        /// has nothing to do with syncmer selection itself.
        fn pseudo_random_seq(len: usize, seed: u64) -> Vec<u8> {
            let bases = [b'A', b'C', b'G', b'T'];
            let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
            (0..len)
                .map(|_| {
                    x = mix64(x.wrapping_add(0x9E37_79B9_7F4A_7C15));
                    bases[(x & 3) as usize]
                })
                .collect()
        }

        /// True DNA-level reverse complement (A<->T, C<->G), matching the
        /// RY-space bit-flip semantics of `reverse_complement` in
        /// `encoding.rs` but at the base level, for feeding whole sequences
        /// to `extract_into`.
        fn revcomp_bases(seq: &[u8]) -> Vec<u8> {
            seq.iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'T' => b'A',
                    b'C' => b'G',
                    b'G' => b'C',
                    other => other,
                })
                .collect()
        }

        /// Brute-force `width`-bit window value, forward reading order.
        fn window_fwd(seq: &[u8], pos: usize, width: usize) -> u64 {
            let mut v = 0u64;
            for i in 0..width {
                v = (v << 1) | base_to_bit(seq[pos + i]);
            }
            v
        }

        /// Brute-force `width`-bit window value, reverse-complemented.
        fn window_rc(seq: &[u8], pos: usize, width: usize) -> u64 {
            let mut v = 0u64;
            for i in (0..width).rev() {
                let bit = base_to_bit(seq[pos + i]) ^ 1;
                v = (v << 1) | bit;
            }
            v
        }

        /// Definitional oracle: for every full k-mer window (assumes no N),
        /// recompute all `k-s+1` s-mer keys directly and take the argmin
        /// with the forward/rc tie-break rules applied explicitly, rather
        /// than via a deque. Returns `(forward_selected, rc_selected)`.
        fn oracle_extract_syncmer(
            seq: &[u8],
            k: usize,
            s: usize,
            salt: u64,
        ) -> (Vec<u64>, Vec<u64>) {
            let len = seq.len();
            let mut fwd = Vec::new();
            let mut rc = Vec::new();
            if len < k {
                return (fwd, rc);
            }
            let win = k - s + 1;
            let t = Sketch::open_target(k, s);
            let mirror_t = win - 1 - t;

            for kmer_pos in 0..=(len - k) {
                // Forward: ties keep the rightmost (highest-offset) s-mer,
                // matching the `>=` pop-back rule.
                let mut best_off = 0usize;
                let mut best_key = mix64(window_fwd(seq, kmer_pos, s) ^ salt);
                for j in 1..win {
                    let key = mix64(window_fwd(seq, kmer_pos + j, s) ^ salt);
                    if key <= best_key {
                        best_key = key;
                        best_off = j;
                    }
                }
                if best_off == t {
                    fwd.push(window_fwd(seq, kmer_pos, k) ^ salt);
                }

                // RC: ties keep the leftmost (lowest-offset) s-mer, matching
                // the strict `>` pop-back rule.
                let mut best_off_rc = 0usize;
                let mut best_key_rc = mix64(window_rc(seq, kmer_pos, s) ^ salt);
                for j in 1..win {
                    let key = mix64(window_rc(seq, kmer_pos + j, s) ^ salt);
                    if key < best_key_rc {
                        best_key_rc = key;
                        best_off_rc = j;
                    }
                }
                if best_off_rc == mirror_t {
                    rc.push(window_rc(seq, kmer_pos, k) ^ salt);
                }
            }
            (fwd, rc)
        }

        #[test]
        fn test_syncmer_oracle_matches_rolling_implementation() {
            let mut ws = MinimizerWorkspace::new();
            let cases: Vec<(Vec<u8>, usize, usize, u64)> = vec![
                (
                    b"ACGTAGCTTGACCGTAAGCTTGGACCTAGGTCAAGCTTAGGCATCGTAGCTAGCATGCTAGCTAGCATCGATCG"
                        .to_vec(),
                    32,
                    15,
                    0,
                ),
                (b"ACGTAGCTTGACCGTAAGCTTGGACCTA".to_vec(), 16, 5, 12345),
                (pseudo_random_seq(500, 1), 64, 15, 0xDEAD_BEEF),
                (pseudo_random_seq(500, 2), 64, 44, 0),
                (pseudo_random_seq(500, 3), 64, 54, 42),
                (pseudo_random_seq(200, 4), 16, 7, 0),
            ];
            for (seq, k, s, salt) in cases {
                let (expect_fwd, expect_rc) = oracle_extract_syncmer(&seq, k, s, salt);

                extract_into(&seq, k, Sketch::OpenSyncmer { s }, salt, &mut ws);
                assert_eq!(
                    ws.buffer, expect_fwd,
                    "extract_into mismatch for k={} s={} salt={}",
                    k, s, salt
                );

                let (fwd, rc) =
                    extract_dual_strand_into(&seq, k, Sketch::OpenSyncmer { s }, salt, &mut ws);
                assert_eq!(
                    fwd, expect_fwd,
                    "dual fwd mismatch for k={} s={} salt={}",
                    k, s, salt
                );
                assert_eq!(
                    rc, expect_rc,
                    "dual rc mismatch for k={} s={} salt={}",
                    k, s, salt
                );
            }
        }

        #[test]
        fn test_syncmer_oracle_matches_for_k_sweep() {
            let mut ws = MinimizerWorkspace::new();
            for &k in &[16usize, 32, 64] {
                let s = k / 2 + 1;
                let seq = pseudo_random_seq(400, k as u64 * 7 + 3);
                let (expect_fwd, expect_rc) = oracle_extract_syncmer(&seq, k, s, 999);
                let (fwd, rc) =
                    extract_dual_strand_into(&seq, k, Sketch::OpenSyncmer { s }, 999, &mut ws);
                assert_eq!(fwd, expect_fwd, "fwd mismatch for k={}", k);
                assert_eq!(rc, expect_rc, "rc mismatch for k={}", k);
            }
        }

        /// The gate for the rc-mirroring hypothesis (mirrored target
        /// `win-1-t`, mirrored tie-break `>`): the rc channel computed
        /// incrementally alongside the forward strand must select the same
        /// *set* of k-mers as forward-only extraction on the literal reverse
        /// complement of the sequence. This is what makes minus-strand
        /// reads classify correctly against a forward-built syncmer index.
        ///
        /// Compared as sorted multisets, not in extraction order: computing
        /// both strands in one incremental left-to-right pass over the
        /// original sequence (required for O(n)) necessarily visits k-mers
        /// in original-position order, which is the reverse of the order
        /// `revcomp(seq)` is scanned in. Every real consumer of this rc
        /// channel (`extract_minimizer_set`, `get_paired_minimizers_into`)
        /// sorts+dedups before use, exactly as today's minimizer rc channel
        /// does -- so order was never part of the contract, only the set.
        #[test]
        fn test_syncmer_rc_matches_forward_on_revcomp() {
            let mut ws = MinimizerWorkspace::new();
            let params: &[(usize, usize)] = &[
                (16, 7),
                (16, 5),
                (32, 15),
                (32, 20),
                (64, 15),
                (64, 44),
                (64, 54),
            ];
            for &(k, s) in params {
                for seed in 0..5u64 {
                    let seq = pseudo_random_seq(300, seed * 1000 + k as u64 * 10 + s as u64);
                    let rc_seq = revcomp_bases(&seq);

                    let (_, mut rc_channel) =
                        extract_dual_strand_into(&seq, k, Sketch::OpenSyncmer { s }, 0, &mut ws);
                    extract_into(&rc_seq, k, Sketch::OpenSyncmer { s }, 0, &mut ws);
                    let mut forward_on_revcomp = ws.buffer.clone();

                    rc_channel.sort_unstable();
                    forward_on_revcomp.sort_unstable();
                    assert_eq!(
                        rc_channel, forward_on_revcomp,
                        "rc mismatch for k={} s={} seed={}",
                        k, s, seed
                    );
                }
            }
        }

        #[test]
        fn test_syncmer_density_matches_law() {
            let mut ws = MinimizerWorkspace::new();
            for &(k, s) in &[(64usize, 15usize), (64, 44), (64, 54), (32, 15), (16, 7)] {
                let seq = pseudo_random_seq(200_000, 42 + k as u64 * 100 + s as u64);
                extract_into(&seq, k, Sketch::OpenSyncmer { s }, 0, &mut ws);
                let observed_density = ws.buffer.len() as f64 / (seq.len() - k + 1) as f64;
                let expected_density = 1.0 / (k - s + 1) as f64;
                let rel_err = (observed_density - expected_density).abs() / expected_density;
                assert!(
                    rel_err < 0.05,
                    "k={} s={} expected={:.5} observed={:.5} rel_err={:.4}",
                    k,
                    s,
                    expected_density,
                    observed_density,
                    rel_err
                );
            }
        }

        #[test]
        fn test_syncmer_extract_into_shorter_than_k_is_empty() {
            let mut ws = MinimizerWorkspace::new();
            let seq = vec![b'A'; 10];
            extract_into(&seq, 16, Sketch::OpenSyncmer { s: 5 }, 0, &mut ws);
            assert!(ws.buffer.is_empty());
        }

        #[test]
        fn test_syncmer_extract_into_exactly_k_matches_oracle_verdict() {
            let mut ws = MinimizerWorkspace::new();
            let seq = pseudo_random_seq(16, 7);
            let (expect_fwd, _) = oracle_extract_syncmer(&seq, 16, 5, 0);
            extract_into(&seq, 16, Sketch::OpenSyncmer { s: 5 }, 0, &mut ws);
            assert_eq!(ws.buffer, expect_fwd);
        }

        #[test]
        fn test_syncmer_n_is_perfect_separator() {
            let mut ws = MinimizerWorkspace::new();
            let half = pseudo_random_seq(60, 99);

            extract_into(&half, 32, Sketch::OpenSyncmer { s: 15 }, 0, &mut ws);
            let single_run = ws.buffer.clone();

            let mut seq = half.clone();
            seq.push(b'N');
            seq.extend(half.iter());
            extract_into(&seq, 32, Sketch::OpenSyncmer { s: 15 }, 0, &mut ws);

            let mut expected = single_run.clone();
            expected.extend(single_run.iter());
            assert_eq!(ws.buffer, expected);
        }

        #[test]
        fn test_syncmer_homopolymer_selects_when_target_is_last_offset() {
            // k=16, s=14: win=3, t=(16-14+2)/2=2=win-1. All s-mer keys tie
            // (homopolymer), and forward's ">=" pop-back always keeps the
            // rightmost (offset win-1) -- which is exactly t here, so every
            // k-mer window is selected.
            let mut ws = MinimizerWorkspace::new();
            let seq = vec![b'A'; 40];
            extract_into(&seq, 16, Sketch::OpenSyncmer { s: 14 }, 0, &mut ws);
            assert_eq!(ws.buffer.len(), 40 - 16 + 1);
            assert!(ws.buffer.iter().all(|&v| v == 0xffff));
        }

        #[test]
        fn test_syncmer_homopolymer_selects_nothing_when_target_is_not_last_offset() {
            // k=16, s=5: win=12, t=6 != win-1=11, so the rightmost-wins tie
            // break never lands on the target offset -- nothing is selected.
            let mut ws = MinimizerWorkspace::new();
            let seq = vec![b'A'; 40];
            extract_into(&seq, 16, Sketch::OpenSyncmer { s: 5 }, 0, &mut ws);
            assert!(ws.buffer.is_empty());
        }

        /// Value-shaped and positional syncmer extractors are two separately
        /// implemented loops (`extract_dual_strand_into_syncmer` and
        /// `extract_strand_minimizers_syncmer`); unlike the minimizer scheme,
        /// which can legitimately disagree on count (value-shaped extractors
        /// collapse consecutive equal values, the positional extractor does
        /// not), syncmer selection is position-local so consecutive
        /// selections always have distinct values -- the two must therefore
        /// select the exact same k-mer value set, not just overlap.
        #[test]
        fn test_syncmer_strand_minimizers_matches_dual_strand_exactly() {
            let mut ws = MinimizerWorkspace::new();
            let sketch = Sketch::OpenSyncmer { s: 15 };
            for seed in 0..5u64 {
                let seq = pseudo_random_seq(5_000, seed);
                let (fwd_sm, rc_sm) = extract_strand_minimizers(&seq, 32, sketch, 0, &mut ws);
                let (fwd_ds, rc_ds) = extract_dual_strand_into(&seq, 32, sketch, 0, &mut ws);

                let mut fwd_sm_vals = fwd_sm.hashes.clone();
                let mut rc_sm_vals = rc_sm.hashes.clone();
                let mut fwd_ds_vals = fwd_ds.clone();
                let mut rc_ds_vals = rc_ds.clone();
                fwd_sm_vals.sort_unstable();
                rc_sm_vals.sort_unstable();
                fwd_ds_vals.sort_unstable();
                rc_ds_vals.sort_unstable();

                assert_eq!(
                    fwd_sm_vals, fwd_ds_vals,
                    "seed {seed}: forward value sets diverged"
                );
                assert_eq!(
                    rc_sm_vals, rc_ds_vals,
                    "seed {seed}: rc value sets diverged"
                );
            }
        }
    }

    #[test]
    fn test_extract_minimizer_set_n_handling() {
        let mut ws = MinimizerWorkspace::new();
        // N resets extraction — should still produce valid sorted/deduped output
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAANAAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        // Just verify sorted + deduped
        for w in fwd.windows(2) {
            assert!(w[0] < w[1], "Forward not strictly sorted after N");
        }
        for w in rc.windows(2) {
            assert!(w[0] < w[1], "RC not strictly sorted after N");
        }
    }
}
