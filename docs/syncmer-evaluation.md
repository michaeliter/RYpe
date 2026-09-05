# Syncmer vs. minimizer sketching: profile results and verdict

Offline evaluation of whether replacing rype's minimizer sketching with syncmer-based sketching
would reduce index size / classification I/O at equal sensitivity. **No production code was
changed for this evaluation.** All numbers below come from a standalone harness,
`examples/sketch_profile.rs`, run against real WoL2 genome and read fixtures. Raw output lives in
`scratch/sketch-profile/`.

## Verdict: **GO**

At both of the user's real operating points (`k=64, w=20` for short reads, `k=64, w=100` for long
reads), open syncmers reach **better-than-iso-sensitivity at roughly 2–2.6x lower selection
density** than rype's current minimizer scheme, and this converts into a genuine **Parquet file-size
reduction of ~45–57%** once tuned past the density rype's own scheme already sits at. This clears
the plan's GO threshold (≥1.3x density ratio, ≥20% net size reduction) by a wide margin. The zero-seed
dead-zone criterion is not independently triggered at the user's actual configured pairings (see
below), but is a serious secondary finding at the mismatched `w=100` + short-read combination.

The cheaper, narrower change — swapping only the *ordering* hash (`min-hash` arm) while keeping
window-minimum selection — is **not** worth doing on its own: it costs real CPU (~35% slower
selection) and measurably *hurts* conservation relative to the current scheme, for a smaller and
less reliable density win than syncmers. If a "go" is scoped down for cost reasons, it should skip
straight to syncmers rather than stop at a real hash.

**Open, not closed.** Both rype usage patterns — single-index host filtration and multi-bucket
taxonomic classification — should use *open* syncmers specifically. Closed syncmers are ruled out at
`w=100` by a hard density floor at `k=64` (they cannot get sparse enough to beat, or even match,
`min-lex`), and at `w=20` they can match today's density but never beat it, unlike open. The one
metric specific to multi-bucket classification (cross-bucket specificity, measured directly — see
below) shows no dependence on scheme choice, so this conclusion holds for both use cases equally;
see "Open vs. closed syncmers" below for the full analysis.

## Method

- Harness: `examples/sketch_profile.rs`, gated by `--self-test`, which must reproduce
  `rype::extract_into` bit-for-bit (the `min-lex` arm calls production code directly) and must match
  each other arm's theoretical density law on random RY sequence before any real measurement is
  trusted. Both self-tests passed on every run in this evaluation.
- Corpus: deterministic 500-genome subset of `perf-data/wol2-genomes` (seed 1, 1,613,052,795 valid
  k-mers), `k=64`, at `w=100` and `w=20`.
- Reads: real `perf-assessment/query-files/short_read_R1.fastq.gz` (200,000 reads) and
  `long_read.fastq.gz` (183,391 reads).
- Mutation model for conservation: transition/transversion-aware (κ=4, i.e. ⅓ of substitutions flip
  the RY bit), θ ∈ {0.5, 1, 2, 5, 10}%, since a uniform-substitution model would understate
  conservation for every arm equally in RY space and make the comparison meaningless.
- **Open/closed parity**: every tested `s` value produces both a `sync-open-s{s}` and a
  `sync-closed-s{s}` arm, so density, conservation, and file size are directly comparable between
  the two kinds at identical `s` — not just "does either reach the target density."
- **Cross-genome specificity** (new for the open-vs-closed / multi-bucket question below): for each
  arm, the 50 conservation-set genomes are paired up adjacent-wise (25 pairs, effectively random
  since the subset order is already seed-shuffled) and the *background* collision rate between
  **unrelated** genomes is measured the same way as conservation, but without mutation. This
  targets the failure mode specific to multi-bucket classification (a spurious cross-bucket seed
  match) which same-genome conservation cannot see.
- Parquet bytes/record: written through the real `ParquetWriteOptions::default()` /
  `to_writer_properties()` path (`src/indices/parquet/options.rs`) — `DELTA_BINARY_PACKED`, Snappy,
  100K-row row groups — not a synthetic estimate. Calibrated against the real single-bucket fixture
  `perf-assessment/config/numerator-w200.ryxdi` (8,000 genomes, `w=200`): that shard measures
  **4.380 B/record** on disk; the harness measures 4.6–5.4 B/record on its synthetic corpora at
  different `w`/arms. Same order of magnitude — the model is a faithful proxy, not a toy. (This
  measures the minimizer column alone, i.e. a single-bucket index; a real multi-bucket index pays a
  scheme-independent extra cost for a non-constant `bucket_id` column on top of this, which does not
  change the open-vs-closed or minimizer-vs-syncmer comparison.)
- Commands, wall time, and peak RSS: recorded via `gtime -v` (macOS lacks GNU `time -v`); logs in
  `scratch/sketch-profile/genomes_wide.log` (14:11.58 wall, 9.60 GB peak RSS — re-run with
  open/closed parity and the specificity metric added) and `scratch/sketch-profile/reads.log`
  (13:47.42 wall, 7.2 MB peak RSS — re-run with open/closed parity).

## Finding 1: rype's minimizer order is lexicographic, not hashed — and it costs conservation

`src/core/extraction.rs` orders candidate k-mers by `kmer ^ salt`. XOR-by-constant is a bijection,
not a mixer, so selection is effectively lexicographic. This has two measured consequences:

**It is denser than a real hash at the same `w`.** At `w=100`: `min-lex` density = 0.02144, vs.
theoretical random-order minimum `2/(w+1)` = 0.01980 (`min-hash`, using a real splitmix64 finalizer
for ordering only, measures 0.01976 — matching theory). At `w=20`: `min-lex` = 0.09547 vs. theory
0.09524 (`min-hash` = 0.09521). `docs/architecture.md:24`'s claim of `|seq|/w` density (i.e. `1/w`)
is off by **~2x** from measured reality at both operating points — the correct floor is `2/(w+1)`,
matching the well-known theoretical minimum for window minimizers, not the doc's naive claim. This
doc line should be corrected as a follow-up.

**A real hash for ordering alone is not a win.** Despite being closer to the density floor,
`min-hash` has *worse* conservation than `min-lex` at every θ tested — e.g. at `w=100, θ=0.01`:
`min-lex` = 0.79944 vs. `min-hash` = 0.76210; at `w=20, θ=0.01`: `min-lex` = 0.80426 vs. `min-hash`
= 0.79848. This is consistent with the literature (structured/biased minimizer orders can
outperform uniform-random order on conservation) and means: **the `min-hash`-only change is a net
negative for this codebase** — it adds a real hash's CPU cost (92–104 Mbase/s vs. 147–150 Mbase/s
selection throughput, a ~35% slowdown) and *loses* sensitivity, for a density gain already
available more cheaply. Not recommended standalone.

## Finding 2: context-independent selection decouples conservation from density

The central result. Both syncmers and `fracmin` (a context-free hash-threshold control) select a
k-mer based only on the k-mer's own content, never on flanking window content. Measured
consequence: conservation is **essentially flat across a >2.5x density range**, while minimizer
conservation is density-coupled and worse at every matched-or-lower density.

**`w=100` (long-read operating point), θ=0.01 conservation vs. selection density:**

| arm | density | conservation | vs. min-lex |
|---|---|---|---|
| `min-lex` (current) | 0.02144 | 0.79944 | baseline |
| `min-hash` | 0.01976 | 0.76210 | worse, cheaper density gain not worth it |
| `sync-open s=15` (density-matched) | 0.02000 | 0.80887 | **higher conservation at lower density** |
| `sync-open s=13` | 0.01921 | 0.80767 | higher conservation, lower density |
| `sync-open s=11` | 0.01850 | 0.80865 | higher conservation, lower density |
| `sync-open s=8` | 0.01754 | 0.80853 | higher conservation, lower density |
| `sync-open s=4` (sparsest testable at k=64) | **0.00811** | **0.80854** | **2.64x lower density, still higher conservation** |
| `fracmin` (θ=1.0, density-matched) | 0.01980 | 0.80846 | higher conservation, lower density |
| `fracmin` (θ=0.5) | 0.00990 | 0.80794 | 2.17x lower density, still higher conservation |

`sync-open` conservation stays essentially pinned at ~0.808 ± 0.001 from s=15 down to s=4 — a
2.64x density range — while `min-lex` sits at 0.799 at nearly 3x the density of `s=4`. The sweep
was widened specifically to find the crossover where syncmers stop beating minimizers; **it was not
found** — s=4 is the sparsest open syncmer expressible at k=64 without the small-alphabet tie
problem (`2^s ≫ k−s+1` requires `s≥4`ish headroom at this window width), so the true
iso-sensitivity crossover is *at least* 2.64x and the harness's k=64 parameterization floor is the
limiting factor, not the syncmer mechanism itself.

**`w=20` (short-read operating point), θ=0.01:**

| arm | density | conservation | vs. min-lex |
|---|---|---|---|
| `min-lex` (current) | 0.09547 | 0.80426 | baseline |
| `min-hash` | 0.09521 | 0.79848 | worse |
| `sync-open s=55` (density-matched) | 0.10000 | 0.80839 | higher conservation, ~matched density |
| `sync-open s=44` (sparsest tested) | **0.04762** | **0.80843** | **2.01x lower density, still higher conservation** |
| `sync-closed s=43` | 0.09092 | 0.80814 | higher conservation, matched density |
| `fracmin` (θ=0.5) | 0.04761 | 0.80871 | 2.01x lower density, still higher conservation |

Same pattern, same floor-limited (not crossover-found) result: ≥2.01x.

## Finding 3: the density win only becomes a file-size win once you go sparser than today's density

Parquet bytes/record *increases* slightly for syncmers at matched density (sparser sorted-integer
sets have larger average deltas, costing more bits under `DELTA_BINARY_PACKED`) — e.g. at `w=100`,
`min-lex` is 4.734 B/rec vs. `sync-open s=15`'s 5.280 B/rec, a 11.5% per-record cost. So net file
size = (record-count ratio) × (bytes/record ratio), and at the density-matched point this is
slightly *negative* (s=15: 1.040x the file size — 4% larger). The win only shows up once density is
cut well past what min-lex already achieves:

| `w=100` arm | file bytes | vs. min-lex (161,626,654 B) |
|---|---|---|
| `sync-open s=15` (matched) | 168,005,698 | +4.0% (worse) |
| `sync-open s=13` | 161,500,199 | −0.1% (wash) |
| `sync-open s=11` | 155,612,019 | −3.7% |
| `sync-open s=8` | 147,691,610 | −8.6% |
| **`sync-open s=4`** | **70,077,666** | **−56.6%** |

| `w=20` arm | file bytes | vs. min-lex (704,289,806 B) |
|---|---|---|
| `sync-open s=55` (matched) | 791,594,457 | +12.4% (worse) |
| `sync-open s=53` | 663,989,303 | −5.7% |
| `sync-open s=48` | 474,551,695 | −32.6% |
| **`sync-open s=44`** | **387,114,060** | **−45.0%** |

Combined with Finding 2 (conservation is flat across this whole range), this means: tuning `s` to
the sparsest value that still avoids the k=64 alphabet floor gives conservation *equal to or better
than* current `min-lex`, at **56.6% smaller index files (w=100)** / **45.0% smaller (w=20)** — both
comfortably clearing the plan's ≥20% net-size-reduction GO bar, on top of clearing the ≥1.3x density
ratio bar by 1.5–2x.

## Open vs. closed syncmers: host filtration (single-index) vs. multi-bucket (taxonomic) classification

Findings 1–3 established that *open* syncmers beat the current minimizer scheme. This section asks
the follow-up question directly: is *closed* a better choice than *open* for either of rype's two
real usage patterns — a single-bucket index used for binary host filtration, or a many-bucket index
used for taxonomic assignment? Answer: **no, for both.** Open dominates closed at both operating
points, and the two use cases don't actually diverge on this question.

### Closed syncmers cannot reach the `w=100` (long-read) target density at `k=64` — ruling them out for host filtration outright

Open and closed syncmers were measured at every tested `s` in parity. For `s ≥ 8` (where
`2^s ≫ k−s+1`, so the small-RY-alphabet tie problem from the plan doesn't bite), closed density
tracks its theoretical `2×` relationship to open almost exactly:

| `s` | open density | closed density | closed / open |
|---|---|---|---|
| 8 | 0.017544 | 0.035673 | 2.03 |
| 11 | 0.018502 | 0.037229 | 2.01 |
| 13 | 0.019211 | 0.038530 | 2.01 |
| 15 | 0.020001 | 0.039996 | 2.00 |

That 2x floor is structural (closed syncmers select 2 of the `k−s+1` possible argmin positions vs.
open's 1), and it means closed syncmers **cannot** reach `w=100`'s target density
(`2/(w+1) = 0.0198`, `min-lex` measures 0.0214) at `k=64` for *any* `s`: even the sparsest tie-safe
setting, `s=8` (0.0357), is 1.7x too dense, and pushing `s` lower to chase a sparser closed syncmer
runs straight into the tie problem — `s=4` measures 0.0622, actually *denser* than `s=8` despite
being intended as sparser, because too few distinct 4-mers (16) exist to break ties among 61
candidate positions without bias. So the closed-syncmer density floor at `k=64` sits well above both
`min-lex`'s density and the target — **not reachable by construction**, independent of which of the
two classification scenarios is in play. Open syncmers have no such floor problem in this range
(Finding 2's `s=4` open value, 0.00811, is real and useful even though it's also past the tie-safe
threshold — bias at `s=4` pushes open density down and closed density up, asymmetrically, which is
itself informative: closed syncmers are structurally the *worse* choice to push toward the sparse
end).

### At `w=20` (short-read), closed reaches parity with today's density but never beats it — while open does

At `w=20`, `min-lex` density (0.0955) is high enough that closed syncmers *can* match it —
`sync-closed-s44` measures 0.09525, matching `min-lex`'s density almost exactly, with conservation
statistically indistinguishable from the density-matched open arm:

| arm | density | conservation @ θ=0.01 |
|---|---|---|
| `min-lex` (current) | 0.09547 | 0.80426 |
| `sync-closed-s44` (density-matched) | 0.09525 | 0.80848 |
| `sync-open-s54` (density-matched) | 0.09091 | 0.80843 |

Both syncmer kinds beat `min-lex` equally at matched density — closed brings no conservation
advantage over open here. But the two diverge sharply once you ask for the density *reduction* that
drove the GO verdict: `sync-open-s44` reaches 0.04762 (2.01x lower than `min-lex`) at conservation
0.80843 — still matching. The closed syncmer at the *same* `s=44` is stuck at 0.09525 (parity with
today, not below it), because closed's 2x-density relationship to open means it would need roughly
double open's effective `s` to reach the same density — `s` values in that range hit the alphabet-tie
problem well before `k=64` allows it. **Closed syncmers can match today's density at `w=20` but
cannot beat it; open can, by 2x.** Since the whole point of this change is a density reduction, closed
is strictly worse here too.

### Cross-genome specificity — the metric that's actually specific to multi-bucket classification — shows no scheme dependence

Same-genome conservation (Findings 1–3) is what matters for single-index host filtration: does a
divergent query still share ≥1 seed with the one reference set? Multi-bucket taxonomic
classification has an extra failure mode conservation can't see: a seed shared **between two
different genomes assigned to different buckets** pollutes the wrong bucket's score. This was
measured directly — background collision rate between 25 pairs of *unrelated* genomes from the
500-genome subset, no mutation:

| arm | `w=100` collision rate | `w=20` collision rate |
|---|---|---|
| `min-lex` | 0.000011 | 0.000010 |
| `min-hash` | 0.000007 | 0.000008 |
| `sync-open` (all `s`) | 0.000005–0.000017 | 0.000010–0.000012 |
| `sync-closed` (all `s`) | 0.000008–0.000017 | 0.000009–0.000011 |
| `fracmin` | 0.000005–0.000006 | 0.000007 |

Every arm sits in the same ~0.00001 band regardless of scheme, density, or open-vs-closed. This is
expected at `k=64`: the chance of two truly unrelated genomes sharing an exact 64-mer by pure chance
is astronomically small, so any cross-genome sharing measured here reflects genuine shared biology
(conserved genes, rRNA, horizontal transfer) sampled at each scheme's own density — not an
artifact of *how* a scheme picks which k-mers to keep. (Caveat: raw shared counts per arm are single
digits to tens out of 25 pairs — a directional read, not a high-powered one — but there is no
systematic trend favoring any scheme, which is itself the informative result.) **Conclusion: scheme
choice (minimizer vs. open vs. closed syncmer) does not create or remove a specificity risk for
multi-bucket classification at `k=64`.** The GO verdict from Findings 1–3 is not undercut by a
hidden cross-bucket cost, and this axis does not favor closed over open either.

### Recommendation for both use cases: open, not closed

For single-index host filtration, closed syncmers are ruled out at `w=100` by a hard density floor
and bring no benefit at `w=20`. For multi-bucket taxonomic classification, the one metric specific to
that scenario (cross-genome specificity) doesn't distinguish open from closed at all, so the
host-filtration conclusion carries over unchanged: **open syncmers are the right choice for both
usage patterns; closed syncmers should not be pursued.**

## Finding 4: warm-up dead zone (secondary GO trigger, not applicable at real operating points)

Minimizer/window schemes require `valid_bases_count ≥ k+w−1` before emitting any output; syncmers
have no such gate. Measured on real reads (`scratch/sketch-profile/reads.tsv`):

- **Short reads @ w=100 (mismatched pairing, not the user's actual config):** `min-lex`/`min-hash`
  = **100% zero-seed** (total classification failure — `k+w-1=163` exceeds most read lengths).
  `sync-open` (s13–15) reduces this to **2.2–2.8%**; `sync-closed` (s13–15) does markedly better
  still, **0.18–0.25%**, roughly 13x lower than open at the same `s` — its ~2x higher density at
  matched `s` (see the open-vs-closed section above) buys real per-read coverage here, since this
  scenario is precisely the one where raw density (not conservation) determines whether a short read
  gets *any* seed at all. `fracmin` is markedly worse at **19.2%** zero-seed despite similar
  corpus-level conservation — confirming that context-independence alone guarantees good *average*
  conservation but not per-window coverage; syncmers' windowed argmin gives both. This is the one
  place closed syncmers have a real, measured edge over open — but it only shows up in a
  misconfigured-`w` scenario the user doesn't actually run, and it doesn't change the recommendation
  above: at both real operating points below, the gap is already negligible for every scheme, so
  there's nothing here for closed to fix.
- **Short reads @ w=20 (user's actual config):** `min-lex` zero-seed = 0.002%, `sync-open` ≈
  0.008–0.030%, `sync-closed` ≈ 0.002% (matching `min-lex`), `fracmin` = 0.107–2.15% — all already
  negligible in absolute terms. No GO trigger here, and no practical reason to prefer closed.
- **Long reads @ w=100 (user's actual config):** all arms ≈ 0.00000–0.00003 zero-seed (`sync-closed`
  hits exactly 0 for `s≥8`) — already negligible for both kinds.

So this finding does not independently trigger GO at the user's real settings, but it is a real,
reportable robustness property: syncmers degrade gracefully under a misconfigured `w`, where the
current scheme fails completely.

## Finding 5: selection throughput is not the bottleneck either way

Single-thread selection throughput (Mbase/s): `min-lex` 147–150, `min-hash`/`sync-open`/`sync-closed`
93–108 (real hashing costs ~30–35%, syncmer argmin costs about the same as `min-hash` — no extra
tax over hashing itself), `fracmin` 470–840 (context-free, no windowing, 4–8x faster). Per
`CLAUDE.md`, classification is I/O-bound on shard loading; even the slowest arm here is far from
being the bottleneck, so a scheme change would not trade I/O savings for a new CPU bottleneck.

## Finding 6: range-based row-group pruning is structurally weak on real indices — bounds how much any density change buys on the I/O path

The plan's "one cheap check": whether `--parallel-rg`'s range-based row-group pruning
(`src/classify/sharded.rs`) already captures most of the possible I/O win, which would cap the
benefit of any density reduction.

**The CLI instrumentation itself is unreachable on both available real fixtures.**
`classify_from_query_index_parallel_rg` (`sharded.rs:696-698`) unconditionally falls back to the
sequential merge-join path whenever `manifest.has_overlapping_shards && shards.len() > 1` — true for
both `n100-w200.ryxdi` and `n97-w50.ryxdi`. The `total_rg_count`/`filtered_rg_count` counters never
fire; `--timing` runs against both fixtures (`scratch/sketch-profile/classify-timing*.log`) confirm
this. Rather than build a new non-overlapping-shard fixture (out of scope), row groups' actual
min/max Parquet statistics were inspected directly via `pyarrow`
(`scratch/sketch-profile/rg_overlap.py`):

| fixture | shard files | row groups | median RG span (frac. of u64) | global range | simulated full-range query keeps |
|---|---|---|---|---|---|
| `n100-w200.ryxdi` | 160 | 5,311 | 0.0008 | 66.7% of u64 | 100% (prunes 0%) |
| `n97-w50.ryxdi` | 11 | 104,656 | 0.0000 | 66.7% of u64 | 100% (prunes 0%) |

Individual row groups are narrow, but their *union* spans exactly 66.7% of the u64 domain on
**both** independently-built fixtures — not a coincidence. `0.667 × u64::MAX` = `0xAAAA...A`, which
is exactly `!salt` for the production default `salt = 0x5555555555555555` (`src/config.rs`).
Reaching a stored value above this requires a k-mer whose XOR with salt exceeds `!salt` — only
possible for a k-mer with an almost perfectly alternating purine/pyrimidine pattern across all 64
bases, vanishingly rare in real sequence — while values *at* the boundary come from a common
low-complexity motif (a 64bp homopolymer/all-purine or all-pyrimidine run). So the same
lexicographic-XOR order from Finding 1 also compresses the practical value range Parquet's
min/max statistics have to work with, on top of not being a real hash for selection purposes.

Net effect: any classify job spanning more than a narrow slice of the corpus's k-mer space (i.e.
any realistic multi-genome or multi-read batch) overlaps nearly every row group's range regardless
of scheme, so range-based pruning is not currently doing meaningful work on real indices and a
sketching-density change would not have to compete with an already-captured I/O win. This confirms
the plan's hypothesis and does not change the verdict.

*(Incidental, minor: confirmed via `rype index stats` that `n100-w200.ryxdi` actually has 160
shards/160 buckets, not "8 shards" as stated in `CLAUDE.md` — worth a doc fix, unrelated to this
evaluation.)*

## What a "go" would cost (scoping only — not implemented here)

Carried from the approved plan, unchanged:

- Selection logic is copy-pasted across three loop bodies (`extract_into`,
  `extract_dual_strand_into`, `extract_strand_minimizers`) — a scheme change must land identically
  in all three or index-side/query-side sketching would silently diverge. First move: factor the
  shared inner loop.
- Add a scheme tag / `s` parameter to `IndexSettings`, CLI args, `ParquetManifest`, and all four
  `k`/`w`/`salt` compatibility validators.
- Bump `FORMAT_VERSION` and tighten `ParquetManifest::load`'s forward-only compatibility check — today
  an old-scheme index loads silently and would produce garbage matches.
- ~30 production call sites thread `(k, w, salt)` positionally; a `SketchParams` struct would make
  this tractable. Plus ~60 test call sites, 4 C FFI entry points, and a `rype.h` regen.
- Density estimators hardcode `1/w` in `src/core/workspace.rs`, `src/memory.rs`,
  `src/classify/sharded.rs` and would need updating to the scheme's real density law.
- `k=64` is required for both operating points (`w=100` needs `1/(k-s+1) ≤ 1/100.5`, unreachable at
  `k=32`) — this is not a new constraint, `w=200` was already `k=64`-only before this evaluation, but
  worth stating explicitly since `w=20` is newly relevant here too.

## Verification

- `target/release/examples/sketch_profile --self-test`: pass (production bit-for-bit match on
  `min-lex`; density-law tolerance checks pass for `min-hash`, `sync-open`, `sync-closed`,
  `fracmin`).
- `cargo test --release`: pass (490+232+58+11+5+23+2+6+43+5 = 875 tests across 14 result lines, 0
  failed, 4+1+2 ignored as expected — no `src/` changes in this evaluation, so this is a regression
  check only).
- `cargo fmt --check`: pass.
- `cargo clippy --all-targets --release`: pass (warnings only, all pre-existing in `src/`/`tests/`
  files this evaluation never touched; none in `examples/sketch_profile.rs`).
- `cargo doc --no-deps --release`: pass.
