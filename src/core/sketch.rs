//! K-mer selection scheme: classic minimizers, or open syncmers.
//!
//! [`Sketch`] occupies the parameter slot formerly held by a bare `w: usize`
//! window size. `k` and `salt` remain separate arguments everywhere in the
//! crate -- they are scheme-independent, and folding them into this type
//! would touch every k/salt-threading call site for no benefit.
//!
//! Minimizer ordering is unchanged, bit-for-bit: minimizers order by raw
//! `kmer ^ salt`. Open syncmers order their s-mers the same way -- raw
//! `smer ^ salt`, unmixed -- per `docs/syncmer-evaluation.md` Finding 8,
//! which found real-hash ordering measurably unnecessary.

use crate::error::RypeError;

/// Which k-mer selection scheme, plus that scheme's single parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sketch {
    /// Classic (w, k)-minimizer: the minimum-hash k-mer in each sliding
    /// window of `w` consecutive k-mer start positions.
    Minimizer { w: usize },
    /// Open syncmer: select a k-mer iff the argmin of its `k - s + 1`
    /// contained s-mers (ordered by raw `smer ^ salt`, same as minimizers)
    /// lands at the conservation-optimal offset `open_target(k, s)`.
    OpenSyncmer { s: usize },
}

/// Serde tag written to the manifest for [`Sketch`]. Absent in a manifest
/// file means `Minimizer` -- this is what lets pre-existing `.ryxdi`
/// indices keep loading without a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SketchSchemeTag {
    #[default]
    Minimizer,
    OpenSyncmer,
}

impl Sketch {
    /// The serde tag naming this sketch's scheme.
    pub fn tag(&self) -> SketchSchemeTag {
        match self {
            Sketch::Minimizer { .. } => SketchSchemeTag::Minimizer,
            Sketch::OpenSyncmer { .. } => SketchSchemeTag::OpenSyncmer,
        }
    }

    /// `w` for minimizers, `0` ("not applicable") for syncmers. `0` is a
    /// safe sentinel because every consumer of it is `estimate_selected`
    /// (which never divides by a syncmer's `w`) or a display path that
    /// gates on `tag()` before printing it.
    pub fn w_or_zero(&self) -> usize {
        match self {
            Sketch::Minimizer { w } => *w,
            Sketch::OpenSyncmer { .. } => 0,
        }
    }

    /// `s` for syncmers, `None` for minimizers.
    pub fn s(&self) -> Option<usize> {
        match self {
            Sketch::Minimizer { .. } => None,
            Sketch::OpenSyncmer { s } => Some(*s),
        }
    }

    /// Target offset for open syncmers: `t = ceil((k - s + 1) / 2)`, the
    /// conservation-optimal offset (Shaw & Yu 2022, Theorem 8).
    #[inline]
    pub fn open_target(k: usize, s: usize) -> usize {
        (k - s + 2) / 2
    }

    /// Inner s-mer window: the number of s-mers contained in one k-mer.
    #[inline]
    pub fn inner_window(k: usize, s: usize) -> usize {
        k - s + 1
    }

    /// Estimated selected k-mers (one strand) in a sequence of `len` bases.
    ///
    /// Minimizer arithmetic (`((len - k) / w + 1) * 2`) is copied verbatim
    /// from the estimators this replaces, so existing buffer/batch sizing
    /// stays bit-identical for minimizer indices. The syncmer arithmetic
    /// uses the measured density law `1/(k - s + 1)` (see
    /// `docs/syncmer-evaluation.md`) -- note this is *not* `((len-k)/w+1)*2`
    /// with some derived `w`; the two schemes have different constant
    /// factors and reusing the minimizer formula would double-count.
    pub fn estimate_selected(&self, len: usize, k: usize) -> usize {
        if len < k {
            return 0;
        }
        match *self {
            Sketch::Minimizer { w } => ((len - k) / w.max(1) + 1) * 2,
            Sketch::OpenSyncmer { s } => (len - k) / Self::inner_window(k, s).max(1) + 1,
        }
    }

    /// Validate `(k, s)` / `(k, w)` for this scheme. `k` is assumed already
    /// checked against `{16, 32, 64}` by the caller (that check is scheme-
    /// independent and lives at the existing call sites).
    pub fn validate(&self, k: usize) -> Result<(), RypeError> {
        match *self {
            Sketch::Minimizer { w } => {
                if w == 0 {
                    return Err(RypeError::validation(
                        "window (w) must be greater than 0 for minimizer sketching",
                    ));
                }
            }
            Sketch::OpenSyncmer { s } => {
                if s == 0 || s >= k {
                    return Err(RypeError::validation(format!(
                        "s must satisfy 0 < s < k for open-syncmer sketching, got s={} k={}",
                        s, k
                    )));
                }
                // Small-alphabet tie warning (locked decision: warn, never
                // reject -- at low s the theoretical density over-estimates
                // the measured one, so downstream buffers over-allocate
                // rather than under-allocate).
                let win = Self::inner_window(k, s);
                if s < 64 && (1u64 << s) < 64 * win as u64 {
                    eprintln!(
                        "warning: s={s} at k={k} gives only {} distinct s-mer values across a \
                         {win}-position window (1-bit RY alphabet); selection may be biased by \
                         tie-breaking rather than value. Recommended s=15 at k=64.",
                        1u64 << s
                    );
                }
            }
        }
        Ok(())
    }

    /// Reconstruct a `Sketch` from manifest fields. `w == 0` sentinel means
    /// "not applicable" and is only legal when `tag` is `OpenSyncmer`.
    pub fn from_fields(
        w: usize,
        tag: SketchSchemeTag,
        s: Option<usize>,
    ) -> Result<Self, RypeError> {
        match tag {
            SketchSchemeTag::Minimizer => {
                if s.is_some() {
                    return Err(RypeError::validation(
                        "manifest is incoherent: scheme=minimizer but s is present",
                    ));
                }
                Ok(Sketch::Minimizer { w })
            }
            SketchSchemeTag::OpenSyncmer => match s {
                Some(s) => Ok(Sketch::OpenSyncmer { s }),
                None => Err(RypeError::validation(
                    "manifest is incoherent: scheme=open_syncmer but s is absent",
                )),
            },
        }
    }

    /// Require two sketches to combine safely (classify, merge, log-ratio,
    /// negative filtering, ...). `ctx` names the operation, for the error.
    ///
    /// This is the single comparison site every combining operation must go
    /// through -- a hand-written `.w() != .w()` check silently ignores a
    /// `scheme`/`s` mismatch (two syncmer indices can share `w = 0` while
    /// using different `s`), degrading a contamination filter into a
    /// silent no-op instead of failing loud.
    pub fn require_compatible(&self, other: &Self, ctx: &str) -> Result<(), RypeError> {
        if self != other {
            return Err(RypeError::validation(format!(
                "{ctx}: incompatible sketch schemes: {self} vs {other}"
            )));
        }
        Ok(())
    }

    /// [`Self::require_compatible`], returning the shared sketch on success
    /// so the caller can extract with it.
    pub fn unify(a: &Self, b: &Self, ctx: &str) -> Result<Self, RypeError> {
        a.require_compatible(b, ctx)?;
        Ok(*a)
    }
}

impl std::fmt::Display for Sketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sketch::Minimizer { w } => write!(f, "minimizer(w={})", w),
            Sketch::OpenSyncmer { s } => write!(f, "open-syncmer(s={})", s),
        }
    }
}

/// Accepts either a bare window size (every existing minimizer call site)
/// or a [`Sketch`] (new scheme-aware call sites). Lets the five public
/// extraction entry points keep their existing `(seq, k, w, salt, ws)`
/// call sites compiling unchanged while new sites pass a scheme explicitly.
pub trait IntoSketch: Copy {
    fn into_sketch(self) -> Sketch;
}

impl IntoSketch for Sketch {
    #[inline]
    fn into_sketch(self) -> Sketch {
        self
    }
}

// Integer impls are load-bearing, not incidental: dozens of existing tests
// call `extract_into(seq, 16, 5, 0, &mut ws)` with unsuffixed integer
// literals. Under a bound like `impl IntoSketch`, an unsuffixed literal's
// inference variable defaults to `i32`; without the `i32` impl those call
// sites stop compiling.
macro_rules! int_into_sketch {
    ($($t:ty),*) => {
        $(
            impl IntoSketch for $t {
                #[inline]
                fn into_sketch(self) -> Sketch {
                    Sketch::Minimizer { w: self as usize }
                }
            }
        )*
    };
}
int_into_sketch!(usize, u32, u64, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_target_matches_shaw_yu() {
        // k=64, s=15: (64 - 15 + 2) / 2 = 25 (integer division)
        assert_eq!(Sketch::open_target(64, 15), 25);
        // k=64, s=4: (64 - 4 + 2) / 2 = 31
        assert_eq!(Sketch::open_target(64, 4), 31);
    }

    #[test]
    fn test_inner_window() {
        assert_eq!(Sketch::inner_window(64, 15), 50);
        assert_eq!(Sketch::inner_window(64, 4), 61);
    }

    #[test]
    fn test_estimate_selected_minimizer_matches_legacy_formula() {
        // Must match `((seq_len - k) / w + 1) * 2` exactly -- this is the
        // formula every existing buffer/batch-size estimator uses today.
        let sketch = Sketch::Minimizer { w: 10 };
        let legacy = ((200 - 32) / 10 + 1) * 2;
        assert_eq!(sketch.estimate_selected(200, 32), legacy);
    }

    #[test]
    fn test_estimate_selected_syncmer_uses_density_law() {
        // Measured law: 1/(k-s+1). For k=64, s=15: inner window = 50.
        let sketch = Sketch::OpenSyncmer { s: 15 };
        let expected = (10_000 - 64) / 50 + 1;
        assert_eq!(sketch.estimate_selected(10_000, 64), expected);
    }

    #[test]
    fn test_estimate_selected_short_sequence_is_zero() {
        assert_eq!(Sketch::Minimizer { w: 50 }.estimate_selected(10, 64), 0);
        assert_eq!(Sketch::OpenSyncmer { s: 15 }.estimate_selected(10, 64), 0);
    }

    #[test]
    fn test_validate_rejects_s_zero() {
        assert!(Sketch::OpenSyncmer { s: 0 }.validate(64).is_err());
    }

    #[test]
    fn test_validate_rejects_s_equal_k() {
        assert!(Sketch::OpenSyncmer { s: 64 }.validate(64).is_err());
    }

    #[test]
    fn test_validate_rejects_s_greater_than_k() {
        assert!(Sketch::OpenSyncmer { s: 100 }.validate(64).is_err());
    }

    #[test]
    fn test_validate_accepts_s_one_below_k() {
        assert!(Sketch::OpenSyncmer { s: 63 }.validate(64).is_ok());
    }

    #[test]
    fn test_validate_low_s_warns_not_rejects() {
        // Locked decision: s=4 at k=64 is measurably tie-biased but must
        // still be accepted, not rejected.
        assert!(Sketch::OpenSyncmer { s: 4 }.validate(64).is_ok());
    }

    #[test]
    fn test_validate_rejects_w_zero() {
        assert!(Sketch::Minimizer { w: 0 }.validate(64).is_err());
    }

    #[test]
    fn test_validate_accepts_normal_minimizer() {
        assert!(Sketch::Minimizer { w: 50 }.validate(64).is_ok());
    }

    #[test]
    fn test_from_fields_minimizer_roundtrip() {
        let sketch = Sketch::from_fields(50, SketchSchemeTag::Minimizer, None).unwrap();
        assert_eq!(sketch, Sketch::Minimizer { w: 50 });
    }

    #[test]
    fn test_from_fields_syncmer_roundtrip() {
        let sketch = Sketch::from_fields(0, SketchSchemeTag::OpenSyncmer, Some(15)).unwrap();
        assert_eq!(sketch, Sketch::OpenSyncmer { s: 15 });
    }

    #[test]
    fn test_from_fields_rejects_minimizer_with_s_present() {
        assert!(Sketch::from_fields(50, SketchSchemeTag::Minimizer, Some(15)).is_err());
    }

    #[test]
    fn test_from_fields_rejects_syncmer_with_s_absent() {
        assert!(Sketch::from_fields(0, SketchSchemeTag::OpenSyncmer, None).is_err());
    }

    #[test]
    fn test_tag_and_accessors() {
        let m = Sketch::Minimizer { w: 50 };
        assert_eq!(m.tag(), SketchSchemeTag::Minimizer);
        assert_eq!(m.w_or_zero(), 50);
        assert_eq!(m.s(), None);

        let s = Sketch::OpenSyncmer { s: 15 };
        assert_eq!(s.tag(), SketchSchemeTag::OpenSyncmer);
        assert_eq!(s.w_or_zero(), 0);
        assert_eq!(s.s(), Some(15));
    }

    #[test]
    fn test_display() {
        assert_eq!(Sketch::Minimizer { w: 50 }.to_string(), "minimizer(w=50)");
        assert_eq!(
            Sketch::OpenSyncmer { s: 15 }.to_string(),
            "open-syncmer(s=15)"
        );
    }

    #[test]
    fn test_scheme_tag_serde_roundtrip() {
        let m = SketchSchemeTag::Minimizer;
        let json = serde_json_like_toml_roundtrip(&m);
        assert_eq!(json, m);

        let s = SketchSchemeTag::OpenSyncmer;
        let json = serde_json_like_toml_roundtrip(&s);
        assert_eq!(json, s);
    }

    #[test]
    fn test_scheme_tag_default_is_minimizer() {
        // This is the backward-compat linchpin: an absent field must
        // deserialize as Minimizer.
        assert_eq!(SketchSchemeTag::default(), SketchSchemeTag::Minimizer);
    }

    #[test]
    fn test_scheme_tag_unknown_string_is_hard_error() {
        // An unrecognized scheme name must fail to parse, not silently
        // default -- this is the one place serde gives free strictness.
        let result: Result<SketchSchemeTag, _> = toml::from_str("value = \"nonsense\"")
            .map(|t: TomlWrapper| t.value)
            .map_err(|e| e.to_string());
        assert!(result.is_err());
    }

    #[derive(serde::Deserialize)]
    struct TomlWrapper {
        value: SketchSchemeTag,
    }

    fn serde_json_like_toml_roundtrip(tag: &SketchSchemeTag) -> SketchSchemeTag {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Wrapper {
            value: SketchSchemeTag,
        }
        let w = Wrapper { value: *tag };
        let s = toml::to_string(&w).unwrap();
        let back: Wrapper = toml::from_str(&s).unwrap();
        back.value
    }

    #[test]
    fn test_into_sketch_integer_literal_is_minimizer() {
        // Mirrors the ~56 existing call sites: extract_into(seq, 16, 5, 0, &mut ws)
        fn accepts(w: impl IntoSketch) -> Sketch {
            w.into_sketch()
        }
        assert_eq!(accepts(5), Sketch::Minimizer { w: 5 });
        assert_eq!(accepts(5u64), Sketch::Minimizer { w: 5 });
        assert_eq!(accepts(5usize), Sketch::Minimizer { w: 5 });
    }

    #[test]
    fn test_into_sketch_sketch_passthrough() {
        fn accepts(w: impl IntoSketch) -> Sketch {
            w.into_sketch()
        }
        let s = Sketch::OpenSyncmer { s: 15 };
        assert_eq!(accepts(s), s);
    }

    #[test]
    fn test_require_compatible_accepts_identical_sketches() {
        let a = Sketch::Minimizer { w: 50 };
        let b = Sketch::Minimizer { w: 50 };
        assert!(a.require_compatible(&b, "test").is_ok());

        let a = Sketch::OpenSyncmer { s: 15 };
        let b = Sketch::OpenSyncmer { s: 15 };
        assert!(a.require_compatible(&b, "test").is_ok());
    }

    #[test]
    fn test_require_compatible_rejects_minimizer_vs_syncmer() {
        let a = Sketch::Minimizer { w: 50 };
        let b = Sketch::OpenSyncmer { s: 15 };
        let err = a.require_compatible(&b, "merge").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("merge"),
            "error should name the context: {msg}"
        );
        assert!(
            msg.contains("minimizer") && msg.contains("open-syncmer"),
            "error should name both schemes: {msg}"
        );
    }

    #[test]
    fn test_require_compatible_rejects_syncmer_different_s() {
        // The exact silent-no-op hazard this helper exists to close: two
        // syncmer indices share the `w = 0` sentinel, so a `.w() != .w()`
        // check alone cannot tell them apart.
        let a = Sketch::OpenSyncmer { s: 15 };
        let b = Sketch::OpenSyncmer { s: 21 };
        assert!(a.require_compatible(&b, "test").is_err());
    }

    #[test]
    fn test_require_compatible_rejects_minimizer_different_w() {
        let a = Sketch::Minimizer { w: 50 };
        let b = Sketch::Minimizer { w: 20 };
        assert!(a.require_compatible(&b, "test").is_err());
    }

    #[test]
    fn test_unify_returns_shared_sketch_on_match() {
        let a = Sketch::OpenSyncmer { s: 15 };
        let b = Sketch::OpenSyncmer { s: 15 };
        assert_eq!(Sketch::unify(&a, &b, "test").unwrap(), a);
    }

    #[test]
    fn test_unify_rejects_mismatch() {
        let a = Sketch::Minimizer { w: 50 };
        let b = Sketch::Minimizer { w: 20 };
        assert!(Sketch::unify(&a, &b, "test").is_err());
    }
}
