//! Shared decode helper for the two on-disk shard row layouts.
//!
//! v1 (`ParquetShardFormat::Parquet`): one row per `(minimizer, bucket_id)`
//! pair — column 1 is a plain `UInt32Array`.
//! v2 (`ParquetShardFormat::Csr`): one row per distinct minimizer — column 1
//! is `List<UInt32>`, one entry per bucket sharing that minimizer.
//!
//! Every shard-reading call site downcasts column 0 (`minimizer`) identically
//! in both formats — only column 1 differs — so this type exists solely to
//! carry that one branch, decided once per `RecordBatch`, instead of matching
//! on format per row.

use arrow::array::{Array, ListArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use std::path::Path;

use super::manifest::ParquetShardFormat;
use crate::error::{Result, RypeError};

/// Decoded bucket-id column of one `RecordBatch`, in whichever physical shape
/// its shard format uses.
pub(crate) enum BucketIdColumn<'a> {
    /// v1: one row per pair.
    Scalar(&'a UInt32Array),
    /// v2: one row per distinct minimizer. `offsets[i]..offsets[i+1]` indexes
    /// into `values` for row `i`'s bucket ids (standard CSR offsets, borrowed
    /// directly from the `ListArray`'s offset buffer — no per-row slicing).
    List {
        offsets: &'a [i32],
        values: &'a UInt32Array,
    },
}

impl<'a> BucketIdColumn<'a> {
    /// Downcast `batch`'s column 1 according to `format`. Column 0
    /// (`minimizer: UInt64`) is unchanged across formats and is downcast
    /// separately by the caller.
    pub(crate) fn downcast(
        batch: &'a RecordBatch,
        format: ParquetShardFormat,
        path: &Path,
    ) -> Result<Self> {
        match format {
            ParquetShardFormat::Parquet => batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .map(BucketIdColumn::Scalar)
                .ok_or_else(|| {
                    RypeError::format(path, "Expected UInt32Array for bucket_id column")
                }),
            ParquetShardFormat::Csr => {
                let list = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| {
                        RypeError::format(path, "Expected List<UInt32> for bucket_ids column")
                    })?;
                let values = list.values().as_any().downcast_ref::<UInt32Array>().ok_or_else(
                    || RypeError::format(path, "Expected UInt32 items in bucket_ids list column"),
                )?;
                Ok(BucketIdColumn::List {
                    offsets: list.offsets(),
                    values,
                })
            }
        }
    }

    /// Number of rows in this column (matches the batch's row count).
    pub(crate) fn len(&self) -> usize {
        match self {
            BucketIdColumn::Scalar(a) => a.len(),
            BucketIdColumn::List { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    /// Push every `(minimizer, bucket_id)` pair carried by row `ri` (whose
    /// minimizer value is `minimizer`) onto `out`. v1: exactly one pair. v2:
    /// one pair per bucket id in that row's list (zero if the list is empty,
    /// which the writer never produces but a corrupt file could).
    pub(crate) fn push_row(&self, ri: usize, minimizer: u64, out: &mut Vec<(u64, u32)>) {
        match self {
            BucketIdColumn::Scalar(a) => out.push((minimizer, a.value(ri))),
            BucketIdColumn::List { offsets, values } => {
                let start = offsets[ri] as usize;
                let end = offsets[ri + 1] as usize;
                out.extend(values.values()[start..end].iter().map(|&b| (minimizer, b)));
            }
        }
    }

    /// Bucket id at position `bi` within row `ri`'s bucket list, or `None`
    /// once `bi` runs past that row's buckets (v1: only `bi == 0` is valid;
    /// v2: `bi` up to that row's list length). Used by readers that walk one
    /// pair at a time (`StreamingShardReader`) rather than expanding a whole
    /// row at once.
    pub(crate) fn bucket_at(&self, ri: usize, bi: usize) -> Option<u32> {
        match self {
            BucketIdColumn::Scalar(a) => (bi == 0).then(|| a.value(ri)),
            BucketIdColumn::List { offsets, values } => {
                let start = offsets[ri] as usize + bi;
                let end = offsets[ri + 1] as usize;
                (start < end).then(|| values.values()[start])
            }
        }
    }
}
