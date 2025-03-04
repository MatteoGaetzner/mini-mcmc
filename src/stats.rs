//! Provides a function to compute the sample covariance matrix for a set of data points.

use nalgebra as na;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use num_traits::Float;
use num_traits::Num;
use simba::scalar::SupersetOf;
use std::error::Error;

pub struct PotentialScaleReduction {
    n: usize,
    mean: Array2<f64>,    // n_chains x n_params
    mean_sq: Array2<f64>, // n_chains x n_params
    n_chains: usize,
    n_params: usize,
}

impl PotentialScaleReduction {
    pub fn new(n_chains: usize, n_params: usize) -> Self {
        let mean_sq = Array2::<f64>::zeros((n_chains, n_params));
        Self {
            n: 0,
            mean: Array2::<f64>::zeros((n_chains, n_params)),
            mean_sq,
            n_chains,
            n_params,
        }
    }

    pub fn step<T>(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: Num
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + std::clone::Clone
            + std::cmp::PartialOrd,
    {
        self.n += 1;

        let n = self.n as f64;
        let x_conv = ndarray::ArrayView2::<T>::from_shape((self.n_chains, self.n_params), x)?
            .mapv(|x| x.to_f64().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_conv.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_conv.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_conv.pow2())) / n;
        };
        Ok(())
    }

    pub fn all(&self) -> Result<Array1<f64>, Box<dyn Error>> {
        let mean_chain = self
            .mean
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean failed.")?;
        let n_chains = self.mean.shape()[0] as f64;
        let n = self.n as f64;
        let fac = n / (n_chains - 1.0);
        let between = (self.mean.clone() - mean_chain.insert_axis(Axis(0)))
            .pow2()
            .sum_axis(Axis(0))
            * fac;
        let sm2 = (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0);
        let within = sm2
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean of squares failed.")?;
        let var = within.clone() * ((n - 1.0) / n) + between * (1.0 / n);
        let rhat = (var / within).sqrt();
        Ok(rhat)
    }

    pub fn max(&self) -> Result<f64, Box<dyn Error>> {
        let all: Array1<f64> = self.all()?;
        let max = *all.max()?;
        Ok(max)
    }
}

/**
Computes the sample covariance matrix from a data matrix `data`, where each row is a
data point and each column is a feature.

# Parameters
- `data`: A `DMatrix<f64>` with shape `(n_rows, n_cols)`, where `n_rows` >= 2.

# Returns
A `DMatrix<f64>` of shape `(n_cols, n_cols)` representing the covariance matrix,
or an error if there are not enough rows.

# Formula
This function centers each column by subtracting its mean, then computes
\[ (Xᵀ X) / n \], where X is the centered data matrix.
*/
pub fn cov<T>(data: &na::DMatrix<T>) -> Result<na::DMatrix<T>, String>
where
    T: Float + na::Field + SupersetOf<f64> + std::clone::Clone + std::fmt::Debug + 'static,
{
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

    let n = T::from(data.nrows()).ok_or(
        format!("Data matrix has too many rows. Couldn't convert usize {:?} to element type of `data`, which has maximal value {:?}", data.nrows(), T::max_value())
    )?;

    // Calculate (X^T * X) / n
    let cov = (centered.transpose() * centered).map(|x| x / n);
    Ok(cov)
}

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;

    #[test]
    fn test_cov_single_row() {
        let data = na::DMatrix::<f64>::from_row_slice(1, 3, &[1_f64, 2_f64, 3_f64]);
        let res = cov(&data);
        assert!(
            res.is_err(),
            "Expected cov(...) with a 1-row matrix to return an error, got {:?}.",
            res
        );
    }

    // Generic helper function to run the Rhat test.
    fn run_rhat_test_generic<T>(data0: Array2<T>, data1: Array2<T>, expected: Array1<f64>, tol: f64)
    where
        T: Float + num_traits::ToPrimitive + num_traits::FromPrimitive + std::fmt::Debug,
        T: ndarray::NdFloat,
    {
        let mut psr = PotentialScaleReduction::new(3, 4);
        psr.step(data0.as_slice().unwrap()).unwrap();
        psr.step(data1.as_slice().unwrap()).unwrap();
        let rhat = psr.all().unwrap();
        let diff = *(rhat.clone() - expected.clone()).abs().max().unwrap();
        assert!(
            diff < tol,
            "Mismatch in Rhat. Got {:?}, expected {:?}, diff = {:?}",
            rhat,
            expected,
            diff
        );
    }

    #[test]
    fn test_rhat_f64_1() {
        // Step 0 data (chains x params)
        let data_step_0 = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);

        // Step 1 data (chains x params)
        let data_step_1 = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f64::consts::SQRT_2, 1.08012345, 0.89442719, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, 1e-7);
    }

    #[test]
    fn test_rhat_f32_1() {
        let data_step_0 = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);
        let data_step_1 = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f64::consts::SQRT_2, 1.0801234, 0.8944271, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, 1e-6);
    }

    #[test]
    fn test_rhat_f32_data() {
        let data_step_0 = arr2(&[
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
        ]);
        let data_step_1 = arr2(&[
            [1.0, 2.0, 0.0, 2.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 2.0],
        ]);
        let expected = array![f64::consts::FRAC_1_SQRT_2, 0.74535599, 1.0, 1.5];
        run_rhat_test_generic(data_step_0, data_step_1, expected, 1e-7);
    }
}
