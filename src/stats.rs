//! Provides a function to compute the sample covariance matrix for a set of data points.

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use num_traits::Num;
use std::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct ChainTracker<T> {
    n_params: usize,
    n: u64,
    p_accept: f32,
    mean: Array1<f32>,    // n_params
    mean_sq: Array1<f32>, // n_params
    last_state: Vec<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainStats {
    pub n: u64,
    pub p_accept: f32,
    pub mean: Array1<f32>, // n_params
    pub sm2: Array1<f32>,  // n_params
}

impl<T: std::clone::Clone> ChainTracker<T> {
    pub fn new(n_params: usize, initial_state: &[T]) -> Self {
        let mean_sq = Array1::<f32>::zeros(n_params);
        let mean = Array1::<f32>::zeros(n_params);
        Self {
            n_params,
            n: 0,
            p_accept: 1.0,
            mean,
            mean_sq,
            last_state: Vec::<T>::from(initial_state),
        }
    }

    pub fn step(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: std::clone::Clone + num_traits::ToPrimitive, //+ num_traits::FromPrimitive , // + std::cmp::PartialOrd,
    {
        self.n += 1;

        // TODO: Update p_accept and last_state

        let n = self.n as f32;
        let x_arr =
            ndarray::ArrayView1::<T>::from_shape(self.n_params, x)?.mapv(|x| x.to_f32().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
        };

        Ok(())
    }

    pub fn sm2(&self) -> Array1<f32> {
        let n = self.n as f32;
        (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0)
    }

    pub fn stats(&self) -> ChainStats {
        ChainStats {
            n: self.n,
            p_accept: self.p_accept,
            mean: self.mean.clone(),
            sm2: self.sm2(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RhatMulti {
    n: usize,
    mean: Array2<f64>,    // n_chains x n_params
    mean_sq: Array2<f64>, // n_chains x n_params
    n_chains: usize,
    n_params: usize,
}

impl RhatMulti {
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
        let x_arr = ndarray::ArrayView2::<T>::from_shape((self.n_chains, self.n_params), x)?
            .mapv(|x| x.to_f64().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
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

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;

    // Generic helper function to run the Rhat test.
    fn run_rhat_test_generic<T>(data0: Array2<T>, data1: Array2<T>, expected: Array1<f64>, tol: f64)
    where
        T: ndarray::NdFloat + num_traits::FromPrimitive,
    {
        let mut psr = RhatMulti::new(3, 4);
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
