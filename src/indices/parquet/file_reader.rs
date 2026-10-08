//! Parquet [`ChunkReader`] for index files.
//!
//! parquet's built-in `impl ChunkReader for File` duplicates the descriptor
//! (`File::try_clone`) on every `get_read` / `get_bytes` call, and Rust's std
//! makes `OwnedFd::try_clone` unconditionally `Unsupported` on wasm32. Opening
//! any index on wasm32-unknown-emscripten therefore failed with "operation not
//! supported on this platform" before a byte was read. [`PositionalFile`] uses
//! positional reads (`pread`) instead, which emscripten supports (it sets
//! `cfg(unix)`; see `advise_prefetch` in `indices::sharded` for the same fact).

/// The chunk reader every parquet reader in this crate is built from
/// (`tests::parquet_readers_use_parquet_file` enforces this).
///
/// Native builds keep parquet's own `File` impl: `PositionalFile::get_bytes`
/// zero-fills its buffer before reading, which measured as ~0.1 s per pass over
/// a 1.7 GB shard, whereas `File`'s `read_to_end` reads into uninitialised capacity.
#[cfg(not(target_os = "emscripten"))]
pub type ParquetFile = std::fs::File;
#[cfg(target_os = "emscripten")]
pub type ParquetFile = PositionalFile;

#[cfg(any(target_os = "emscripten", test))]
pub use positional::PositionalFile;

// Only wired in on emscripten; compiled under `test` so the native unit test runs.
#[cfg(any(target_os = "emscripten", test))]
mod positional {
    use bytes::Bytes;
    use parquet::errors::Result as ParquetResult;
    use parquet::file::reader::{ChunkReader, Length};
    use std::fs::File;
    use std::io::{self, BufReader, Read};
    use std::os::unix::fs::FileExt;
    use std::path::Path;
    use std::sync::Arc;

    /// File-backed parquet chunk reader using positional reads (no `try_clone`).
    pub struct PositionalFile {
        file: Arc<File>,
        len: u64,
    }

    impl PositionalFile {
        /// Open `path` for reading and record its length.
        pub fn open(path: &Path) -> io::Result<Self> {
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            Ok(Self {
                file: Arc::new(file),
                len,
            })
        }
    }

    impl Length for PositionalFile {
        fn len(&self) -> u64 {
            self.len
        }
    }

    /// Sequential `Read` over a shared handle, advancing by `pread`.
    pub struct PreadReader {
        file: Arc<File>,
        pos: u64,
    }

    impl Read for PreadReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.file.read_at(buf, self.pos)?;
            self.pos += n as u64;
            Ok(n)
        }
    }

    impl ChunkReader for PositionalFile {
        // parquet calls `get_read` for the footer and once per page header (a few
        // dozen bytes each); a small buffer avoids re-reading page data that
        // `get_bytes` fetches right after.
        type T = BufReader<PreadReader>;

        fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
            Ok(BufReader::with_capacity(
                1024,
                PreadReader {
                    file: Arc::clone(&self.file),
                    pos: start,
                },
            ))
        }

        fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
            let mut buf = vec![0u8; length];
            self.file.read_exact_at(&mut buf, start)?;
            Ok(buf.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{UInt32Array, UInt64Array};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use parquet::file::reader::{ChunkReader, Length};
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    use std::sync::Arc;

    /// `PositionalFile` must give the parquet reader exactly what was written:
    /// same footer, same pages, same rows. Multiple row groups and pages exercise
    /// both `get_read` (page headers) and `get_bytes` (page data). Runs on wasm32
    /// too (`cargo test --target wasm32-unknown-emscripten`), where `File` itself
    /// cannot serve as a reference.
    #[test]
    fn positional_file_reads_parquet_identically_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.parquet");
        let n = 50_000u64;
        let batch = RecordBatch::try_from_iter([
            (
                "minimizer",
                Arc::new(UInt64Array::from_iter_values(0..n)) as arrow::array::ArrayRef,
            ),
            (
                "bucket_id",
                Arc::new(UInt32Array::from_iter_values((0..n).map(|v| v as u32 % 7))) as _,
            ),
        ])
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(10_000))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let positional = PositionalFile::open(&path).unwrap();
        assert_eq!(positional.len(), std::fs::metadata(&path).unwrap().len());
        let batches: Vec<RecordBatch> = ParquetRecordBatchReaderBuilder::try_new(positional)
            .unwrap()
            .build()
            .unwrap()
            .map(|b| b.unwrap())
            .collect();
        assert!(batches.len() > 1, "expected several row groups to be read");
        let read_back = arrow::compute::concat_batches(&batch.schema(), &batches).unwrap();
        assert_eq!(read_back, batch);

        // The reader contract past EOF: get_bytes errors, get_read yields what is left.
        let positional = PositionalFile::open(&path).unwrap();
        assert!(positional.get_bytes(positional.len() - 4, 8).is_err());
        let mut tail = Vec::new();
        positional
            .get_read(positional.len() - 4)
            .unwrap()
            .read_to_end(&mut tail)
            .unwrap();
        assert_eq!(tail, b"PAR1");
    }

    /// Any `std::fs::File` handed to a parquet reader silently breaks wasm32
    /// (see module doc). Every parquet-reading site under `src/indices` and
    /// `src/memory.rs` must open through `ParquetFile`; a `File::open` there is
    /// only allowed with a `not a parquet reader` comment within the two lines above it.
    /// Scans the source tree, so it is host-only (no sources on an emscripten test run).
    #[test]
    #[cfg(not(target_os = "emscripten"))]
    fn parquet_readers_use_parquet_file() {
        fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|e| e == "rs") {
                    out.push(p);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = vec![root.join("memory.rs")];
        walk(&root.join("indices"), &mut files);
        let mut offenders = Vec::new();
        for file in files {
            if file.ends_with("file_reader.rs") {
                continue;
            }
            let src = std::fs::read_to_string(&file).unwrap();
            let lines: Vec<&str> = src.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let marked = lines[i.saturating_sub(2)..=i]
                    .iter()
                    .any(|l| l.contains("not a parquet reader"));
                if line.contains("File::open(") && !line.contains("ParquetFile::open(") && !marked {
                    offenders.push(format!("{}:{}", file.display(), i + 1));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "File::open fed to parquet? use ParquetFile::open: {offenders:?}"
        );
    }
}
