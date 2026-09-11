//! A parquet [`ChunkReader`] over a file that does not depend on `File::try_clone`.
//!
//! parquet's built-in `impl ChunkReader for File` duplicates the descriptor on
//! every `get_read` / `get_bytes` call. Rust's std makes `OwnedFd::try_clone`
//! unconditionally return `Unsupported` on wasm32 (including
//! `wasm32-unknown-emscripten`), so opening any index there failed with
//! "operation not supported on this platform" before a single byte was read.
//!
//! Positional reads (`pread` / `seek_read`) are available on every platform rype
//! builds for, so this reader uses them for `get_bytes` and re-opens the path
//! for the (rare, footer / page-stream) `get_read` calls.
//!
//! Native builds keep parquet's own `File` reader: [`PositionalFile::get_bytes`]
//! zero-fills each buffer before reading, which measured as ~0.1 s per pass over
//! a 1.7 GB shard, whereas `File`'s `read_to_end` path reads into uninitialised
//! capacity. [`ParquetFile`] selects the right type per target.

use bytes::Bytes;
use parquet::errors::Result as ParquetResult;
use parquet::file::reader::{ChunkReader, Length};
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Parquet chunk reader used for index files: `File` natively, positional reads on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub type ParquetFile = File;
#[cfg(target_arch = "wasm32")]
pub type ParquetFile = PositionalFile;

/// File-backed parquet chunk reader using positional reads (no `try_clone`).
// Only wired in on wasm32; kept compiled natively so its unit test runs.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub struct PositionalFile {
    path: PathBuf,
    file: File,
    len: u64,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl PositionalFile {
    /// Open `path` for reading and record its length.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            len,
        })
    }

    #[cfg(unix)]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.file.seek_read(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ))
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl Length for PositionalFile {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for PositionalFile {
    type T = BufReader<File>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(start))?;
        Ok(BufReader::new(file))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let mut buf = vec![0u8; length];
        self.read_exact_at(&mut buf, start)?;
        Ok(buf.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn positional_and_streaming_reads_match_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        File::create(&path).unwrap().write_all(&data).unwrap();

        let reader = PositionalFile::open(&path).unwrap();
        assert_eq!(reader.len(), 1000);
        assert_eq!(&reader.get_bytes(10, 20).unwrap()[..], &data[10..30]);

        let mut tail = Vec::new();
        reader
            .get_read(990)
            .unwrap()
            .read_to_end(&mut tail)
            .unwrap();
        assert_eq!(&tail[..], &data[990..]);

        // Reading past EOF is an error, matching parquet's File impl.
        assert!(reader.get_bytes(995, 10).is_err());
    }
}
