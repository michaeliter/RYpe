//! Fixtures shared by the Arrow integration test binaries.

/// Deterministic pseudo-DNA. Distinct `seed`s give distinct minimizer sets.
///
/// A period-4 pattern (`bases[(i + seed) % 4]`) will not do: every seed is then
/// a rotation of `ACGT...`, so all references and queries share one minimizer
/// set and every hit scores 1.0. Under such a fixture the classification
/// threshold is dead weight — a classifier that ignored it entirely passes.
///
/// Kept here rather than copied per test binary so a change to the generator
/// (widening the k-mer space, say) cannot leave the fixtures disagreeing.
/// `src/arrow/mod.rs` keeps its own copy because unit tests compile into the
/// library and cannot reach `tests/`.
pub fn generate_sequence(len: usize, seed: u64) -> Vec<u8> {
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
