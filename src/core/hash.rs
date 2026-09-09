//! Order-mixing hash, test-only.
//!
//! Open-syncmer s-mer selection originally ordered by `mix64(smer ^ salt)`;
//! `docs/syncmer-evaluation.md` Finding 8 found real-hash ordering measurably
//! unnecessary (a synthetic-mutation conservation sweep on two real genome
//! corpora, bit-for-bit production oracle/RC-mirroring tests, and real
//! cross-individual read recall all showed lex ordering -- the same raw
//! `smer ^ salt` minimizers already use, unmixed -- performs identically
//! while recovering the hashing cost), so production now orders s-mers the
//! same way minimizers do. `mix64` remains as a deterministic,
//! dependency-free pseudo-random sequence generator for test fixtures
//! (e.g. `classify::sharded`'s tests) -- a bijection is a convenient
//! avalanche step for that, unrelated to its former sketching role.

/// splitmix64 finalizer (Steele, Lea & Flood 2014; the mixing stage of
/// `next()` in the reference `splitmix64.c` generator, without the golden-
/// gamma increment). Not a general-purpose hash: it is a bijection on `u64`,
/// used here as a test-fixture avalanche step (see module doc comment).
#[cfg(test)]
#[inline(always)]
pub(crate) fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vectors computed independently in Python (not derived from
    /// this Rust implementation), pinning the exact splitmix64 constants.
    /// A wrong constant, a dropped `^=`, or a swapped shift amount changes
    /// every one of these.
    #[test]
    fn test_mix64_known_vectors() {
        let vectors: &[(u64, u64)] = &[
            (0x0, 0x0),
            (0x1, 0x5692161d100b05e5),
            (0x2, 0xdbd238973a2b148a),
            (0x2a, 0xa759ea27d4727622),
            (0xdeadbeef, 0x4e062702ec929eea),
            (0xffffffffffffffff, 0xb4d055fcf2cbbd7b),
            (0x123456789abcdef0, 0x9629f58e8ec5b906),
        ];
        for &(input, expected) in vectors {
            assert_eq!(
                mix64(input),
                expected,
                "mix64({:#x}) mismatch: got {:#x}, want {:#x}",
                input,
                mix64(input),
                expected
            );
        }
    }

    /// splitmix64's finalizer is a bijection on u64 (it's built from
    /// invertible xor-shift/multiply steps). Over any sample of distinct
    /// inputs, outputs must also be distinct -- a collision here would mean
    /// the mixer degenerated, which would silently bias which s-mer wins
    /// ties instead of just picking arbitrarily among them.
    #[test]
    fn test_mix64_bijective_spot_check() {
        let mut seen_inputs = std::collections::HashSet::with_capacity(20_000);
        let mut seen_outputs = std::collections::HashSet::with_capacity(20_000);
        let mut check = |x: u64| {
            // Only a genuinely new input can expose a bijection violation --
            // re-testing the same x twice would trivially "collide" with
            // itself without saying anything about the mixer.
            if seen_inputs.insert(x) {
                assert!(seen_outputs.insert(mix64(x)), "collision at x={:#x}", x);
            }
        };
        for x in 0..10_000u64 {
            check(x);
        }
        // Also check a scattered high-bit range, not just small integers.
        for i in 0..10_000u64 {
            let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            check(x);
        }
    }
}
