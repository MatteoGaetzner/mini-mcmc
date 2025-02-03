//! Provides a function to compute the sample covariance matrix for a set of data points.

use nalgebra as na;

/// Computes the sample covariance matrix from a data matrix `data`, where each row is a
/// data point and each column is a feature.
///
/// # Parameters
/// - `data`: A `DMatrix<f64>` with shape `(n_rows, n_cols)`, where `n_rows` >= 2.
///
/// # Returns
/// A `DMatrix<f64>` of shape `(n_cols, n_cols)` representing the covariance matrix,
/// or an error if there are not enough rows.
///
/// # Formula
/// This function centers each column by subtracting its mean, then computes
/// \[ (Xᵀ X) / n \], where X is the centered data matrix.
pub fn cov(data: &na::DMatrix<f64>) -> Result<na::DMatrix<f64>, String> {
    if data.nrows() <= 1 {
        return Err(format!(
            "Expected matrix to have at least 2 rows but it has {}",
            data.nrows()
        ));
    }

    let mut centered = data.clone();
    for (mut col, &mean) in centered.column_iter_mut().zip(data.row_mean().iter()) {
        col.add_scalar_mut(-mean);
    }

    let n = data.nrows();
    // Calculate (X^T * X) / n
    let cov = (centered.transpose() * centered).map(|x| x / (n as f64));
    Ok(cov)
}
