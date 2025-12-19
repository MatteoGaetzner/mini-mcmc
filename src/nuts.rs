//! Burn-backed NUTS wrapper over the backend-agnostic core.

use crate::distributions::GradientTarget;
use crate::generic_hmc::HamiltonianTarget;
use crate::generic_nuts::{GenericNUTS, GenericNUTSChain};
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Element, ElementConversion, Tensor};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::distr::Distribution as RandDistribution;
// Bind to rand's Distribution to avoid mismatches from transitive rand 0.8 deps.
use rand_distr::uniform::SampleUniform;
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use std::error::Error;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
struct BurnGradientTarget<GTarget> {
    inner: GTarget,
}

impl<T, B, GTarget> HamiltonianTarget<Tensor<B, 1>> for BurnGradientTarget<GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: GradientTarget<T, B> + Sync,
    StandardNormal: RandDistribution<T>,
{
    fn logp_and_grad(&self, position: &Tensor<B, 1>, grad: &mut Tensor<B, 1>) -> T {
        let (logp, grad_tensor) = self.inner.unnorm_logp_and_grad(position.clone());
        grad.inplace(|_| grad_tensor.clone());
        logp.into_scalar()
    }
}

/// Burn-backed No-U-Turn Sampler (NUTS).
#[derive(Clone)]
pub struct NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + ToPrimitive
        + Copy
        + Send,
    B: AutodiffBackend<FloatElem = T> + Send,
    GTarget: GradientTarget<T, B> + Sync + Clone + Send,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
    Exp1: RandDistribution<T>,
{
    inner: GenericNUTS<Tensor<B, 1>, BurnGradientTarget<GTarget>>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + ToPrimitive
        + Copy
        + Send,
    B: AutodiffBackend<FloatElem = T> + Send,
    GTarget: GradientTarget<T, B> + Sync + Clone + Send,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
    Exp1: RandDistribution<T>,
{
    pub fn new(target: GTarget, initial_positions: Vec<Vec<T>>, target_accept_p: T) -> Self {
        let positions_vec: Vec<Tensor<B, 1>> = initial_positions
            .into_iter()
            .map(|pos| {
                let len = pos.len();
                let td: TensorData = TensorData::new(pos, [len]);
                Tensor::<B, 1>::from_data(td, &B::Device::default())
            })
            .collect();
        let inner = GenericNUTS::new(
            BurnGradientTarget {
                inner: target.clone(),
            },
            positions_vec,
            target_accept_p,
        );
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        let sample = self.inner.run(n_collect, n_discard);
        array3_to_tensor(sample)
    }

    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        let (sample, stats) = self.inner.run_progress(n_collect, n_discard)?;
        Ok((array3_to_tensor(sample), stats))
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.set_seed(seed);
        self
    }
}

/// Burn-backed single-chain NUTS wrapper.
#[derive(Clone)]
pub struct NUTSChain<T, B, GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + ToPrimitive + Copy,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: GradientTarget<T, B> + Sync + Clone,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
    Exp1: RandDistribution<T>,
{
    inner: GenericNUTSChain<Tensor<B, 1>, BurnGradientTarget<GTarget>>,
    pub position: Tensor<B, 1>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTSChain<T, B, GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + ToPrimitive + Copy,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: GradientTarget<T, B> + Sync + Clone,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
    Exp1: RandDistribution<T>,
{
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: T) -> Self {
        let len = initial_position.len();
        let td: TensorData = TensorData::new(initial_position, [len]);
        let position = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let inner = GenericNUTSChain::new(
            BurnGradientTarget { inner: target },
            position.clone(),
            target_accept_p,
        );
        Self {
            inner,
            position,
            _phantom: PhantomData,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.set_seed(seed);
        self.position = self.inner.position().clone();
        self
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        let sample = self.inner.run(n_collect, n_discard);
        self.position = self.inner.position().clone();
        array2_to_tensor(sample)
    }

    pub fn step(&mut self) {
        self.inner.step();
        self.position = self.inner.position().clone();
    }
}

fn array2_to_tensor<B, T>(arr: ndarray::Array2<T>) -> Tensor<B, 2>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    let shape = arr.raw_dim();
    let (mut data, offset) = arr.into_raw_vec_and_offset();
    if let Some(offset) = offset {
        if offset != 0 {
            data.rotate_left(offset);
        }
    }
    let td = TensorData::new(data, [shape[0], shape[1]]);
    Tensor::<B, 2>::from_data(td, &B::Device::default())
}

fn array3_to_tensor<B, T>(arr: ndarray::Array3<T>) -> Tensor<B, 3>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    let shape = arr.raw_dim();
    let (mut data, offset) = arr.into_raw_vec_and_offset();
    if let Some(offset) = offset {
        if offset != 0 {
            data.rotate_left(offset);
        }
    }
    let td = TensorData::new(data, [shape[0], shape[1], shape[2]]);
    Tensor::<B, 3>::from_data(td, &B::Device::default())
}
