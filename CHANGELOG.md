# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--dedup-cross-bucket` / `--dedup-cross-bucket-min` opt-in index-build option (`index create` and `index from-config`) to remove minimizers shared across too many buckets, improving discrimination between closely related genomes at some sensitivity cost. See `docs/architecture.md` for details and measured trade-offs.
- Initial public release
- RY-space minimizer-based sequence classification
- Support for k-mer sizes 16, 32, and 64
- Single-end and paired-end read classification
- Sharded index support for large datasets
- Inverted index for memory-efficient classification
- C API for FFI integration
- CLI tool for index management and classification

### Changed

### Deprecated

### Removed

### Fixed

### Security
