//! Numerical parity metrics shared by scalar and accelerated backends.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::{backend::TensorBackend, tensor::Tensor};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParityMetrics {
    pub shape: Vec<usize>,
    pub elements: usize,
    pub cosine_similarity: f64,
    pub maximum_absolute_error: f32,
    pub mean_absolute_error: f64,
    pub actual_nan_count: usize,
    pub expected_nan_count: usize,
    pub actual_infinity_count: usize,
    pub expected_infinity_count: usize,
}

impl ParityMetrics {
    pub fn has_non_finite_values(&self) -> bool {
        self.actual_nan_count != 0
            || self.expected_nan_count != 0
            || self.actual_infinity_count != 0
            || self.expected_infinity_count != 0
    }

    pub fn require(&self, tolerance: ParityTolerance) -> Result<()> {
        if self.has_non_finite_values() {
            bail!(
                "parity comparison contains non-finite values: actual nan={} inf={}, expected nan={} inf={}",
                self.actual_nan_count,
                self.actual_infinity_count,
                self.expected_nan_count,
                self.expected_infinity_count
            );
        }
        if self.cosine_similarity < tolerance.minimum_cosine_similarity {
            bail!(
                "cosine similarity {:.9} is below {:.9}",
                self.cosine_similarity,
                tolerance.minimum_cosine_similarity
            );
        }
        if self.maximum_absolute_error > tolerance.maximum_absolute_error {
            bail!(
                "maximum absolute error {:.9} exceeds {:.9}",
                self.maximum_absolute_error,
                tolerance.maximum_absolute_error
            );
        }
        if self.mean_absolute_error > tolerance.maximum_mean_absolute_error {
            bail!(
                "mean absolute error {:.9} exceeds {:.9}",
                self.mean_absolute_error,
                tolerance.maximum_mean_absolute_error
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParityTolerance {
    pub minimum_cosine_similarity: f64,
    pub maximum_absolute_error: f32,
    pub maximum_mean_absolute_error: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct BackendParity {
    pub reference_backend: &'static str,
    pub candidate_backend: &'static str,
    pub reference_runtime: Duration,
    pub candidate_runtime: Duration,
    pub metrics: ParityMetrics,
}

pub(crate) fn compare_tensors(actual: &Tensor, expected: &Tensor) -> Result<ParityMetrics> {
    if actual.shape() != expected.shape() {
        bail!(
            "parity shape mismatch: actual {:?}, expected {:?}",
            actual.shape(),
            expected.shape()
        );
    }

    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut maximum_absolute_error = 0.0f32;
    let mut absolute_error_sum = 0.0f64;
    let mut actual_nan_count = 0usize;
    let mut expected_nan_count = 0usize;
    let mut actual_infinity_count = 0usize;
    let mut expected_infinity_count = 0usize;

    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        actual_nan_count += usize::from(actual.is_nan());
        expected_nan_count += usize::from(expected.is_nan());
        actual_infinity_count += usize::from(actual.is_infinite());
        expected_infinity_count += usize::from(expected.is_infinite());
        if !actual.is_finite() || !expected.is_finite() {
            continue;
        }
        let error = (actual - expected).abs();
        maximum_absolute_error = maximum_absolute_error.max(error);
        absolute_error_sum += error as f64;
        dot += actual as f64 * expected as f64;
        actual_norm += actual as f64 * actual as f64;
        expected_norm += expected as f64 * expected as f64;
    }

    let has_non_finite_values = actual_nan_count != 0
        || expected_nan_count != 0
        || actual_infinity_count != 0
        || expected_infinity_count != 0;
    let elements = actual.len();
    let cosine_similarity = if has_non_finite_values {
        f64::NAN
    } else if actual_norm == 0.0 || expected_norm == 0.0 {
        if actual_norm == expected_norm {
            1.0
        } else {
            0.0
        }
    } else {
        dot / (actual_norm.sqrt() * expected_norm.sqrt())
    };

    Ok(ParityMetrics {
        shape: actual.shape().to_vec(),
        elements,
        cosine_similarity,
        maximum_absolute_error: if has_non_finite_values {
            f32::INFINITY
        } else {
            maximum_absolute_error
        },
        mean_absolute_error: if has_non_finite_values {
            f64::INFINITY
        } else if elements == 0 {
            0.0
        } else {
            absolute_error_sum / elements as f64
        },
        actual_nan_count,
        expected_nan_count,
        actual_infinity_count,
        expected_infinity_count,
    })
}

pub(crate) fn compare_backends<F>(
    reference: &dyn TensorBackend,
    candidate: &dyn TensorBackend,
    execute: F,
) -> Result<BackendParity>
where
    F: Fn(&dyn TensorBackend) -> Result<Tensor>,
{
    let started = Instant::now();
    let expected = execute(reference)?;
    let reference_runtime = started.elapsed();
    let started = Instant::now();
    let actual = execute(candidate)?;
    let candidate_runtime = started.elapsed();
    let metrics = compare_tensors(&actual, &expected)?;
    Ok(BackendParity {
        reference_backend: reference.name(),
        candidate_backend: candidate.name(),
        reference_runtime,
        candidate_runtime,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SCALAR_BACKEND;

    #[test]
    fn awkward_shape_backend_add_has_exact_parity() {
        let left = Tensor::new(
            vec![2, 3, 5],
            (0..30).map(|index| index as f32 * 0.125 - 1.5).collect(),
        )
        .unwrap();
        let right = Tensor::new(
            vec![2, 3, 5],
            (0..30)
                .map(|index| ((index * 7) % 13) as f32 * -0.0625)
                .collect(),
        )
        .unwrap();
        let parity = compare_backends(&SCALAR_BACKEND, &SCALAR_BACKEND, |backend| {
            backend.add(&left, &right)
        })
        .unwrap();
        assert_eq!(parity.reference_backend, "scalar-cpu");
        assert_eq!(parity.candidate_backend, "scalar-cpu");
        assert!(parity.reference_runtime <= Duration::from_secs(1));
        assert!(parity.candidate_runtime <= Duration::from_secs(1));
        assert_eq!(parity.metrics.shape, [2, 3, 5]);
        assert_eq!(parity.metrics.elements, 30);
        assert_eq!(parity.metrics.cosine_similarity, 1.0);
        assert_eq!(parity.metrics.maximum_absolute_error, 0.0);
        assert_eq!(parity.metrics.mean_absolute_error, 0.0);
        parity
            .metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
    }

    #[test]
    fn reports_shape_mismatch_before_comparing_values() {
        let actual = Tensor::zeros(vec![2, 3]).unwrap();
        let expected = Tensor::zeros(vec![3, 2]).unwrap();
        assert!(compare_tensors(&actual, &expected).is_err());
    }

    #[test]
    fn counts_nan_and_infinity_and_rejects_them() {
        let actual = Tensor::new(vec![4], vec![1.0, f32::NAN, f32::INFINITY, 4.0]).unwrap();
        let expected = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, f32::NEG_INFINITY]).unwrap();
        let metrics = compare_tensors(&actual, &expected).unwrap();
        assert_eq!(metrics.actual_nan_count, 1);
        assert_eq!(metrics.expected_nan_count, 0);
        assert_eq!(metrics.actual_infinity_count, 1);
        assert_eq!(metrics.expected_infinity_count, 1);
        assert!(metrics.cosine_similarity.is_nan());
        assert!(metrics.maximum_absolute_error.is_infinite());
        assert!(metrics.mean_absolute_error.is_infinite());
        assert!(
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.0,
                    maximum_absolute_error: f32::INFINITY,
                    maximum_mean_absolute_error: f64::INFINITY,
                })
                .is_err()
        );
    }
}
