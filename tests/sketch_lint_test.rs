//! Source-level lint: no hand-written `w`-field compatibility comparison
//! outside `core/sketch.rs`.
//!
//! `w` is meaningless for open-syncmer indices (stored as the sentinel `0`),
//! so a bare `a.w() != b.w()` silently treats any two syncmer indices as
//! compatible regardless of `s` -- this is exactly the bug class Phase 6
//! closed by routing every combining operation (merge, log-ratio, negative
//! filtering, ...) through `Sketch::require_compatible`/`Sketch::unify`.
//! This test is the guard against a *ninth* combining operation being added
//! later with a hand-rolled `w` check instead of using those helpers.
//!
//! A line legitimately comparing `w` against something other than another
//! index's `w` (a sentinel, a literal) is not the risk this test guards
//! against -- mark it with a `// scheme-blind:` comment on the line above
//! to exempt it (see `src/indices/inverted/mod.rs`'s dead
//! `InvertedIndex::validate_against_metadata` for the one existing case:
//! that struct predates `Sketch` and has no `sketch` field to compare).

use std::path::Path;

const RISKY_PATTERNS: &[&str] = &[".w() != ", ".w != ", ".w() ==", ".w =="];

/// Recursively collect `.rs` files under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir should succeed for src/") {
        let entry = entry.expect("dir entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn test_no_hand_written_w_comparison_outside_sketch_rs() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest_dir).join("src");

    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    assert!(!files.is_empty(), "expected to find .rs files under src/");

    let sketch_rs = src_dir.join("core").join("sketch.rs");

    let mut violations = Vec::new();
    for file in &files {
        if file == &sketch_rs {
            continue;
        }
        let contents = std::fs::read_to_string(file).expect("source file should be readable");
        let lines: Vec<&str> = contents.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // Strip line comments so a pattern only mentioned in prose
            // (like this file's own doc comment) doesn't self-trigger.
            let code = line.split("//").next().unwrap_or("");
            if !RISKY_PATTERNS.iter().any(|p| code.contains(p)) {
                continue;
            }
            let exempted = i > 0 && lines[i - 1].contains("scheme-blind:");
            if exempted {
                continue;
            }
            violations.push(format!("{}:{}: {}", file.display(), i + 1, line.trim()));
        }
    }

    assert!(
        violations.is_empty(),
        "Found hand-written `w`-comparison(s) outside core/sketch.rs -- use \
         Sketch::require_compatible/Sketch::unify instead, so a scheme/`s` \
         mismatch (two syncmer indices sharing the `w = 0` sentinel) isn't \
         silently ignored. If this comparison genuinely isn't a cross-index \
         compatibility check, add a `// scheme-blind:` comment on the \
         preceding line explaining why.\n{}",
        violations.join("\n")
    );
}
