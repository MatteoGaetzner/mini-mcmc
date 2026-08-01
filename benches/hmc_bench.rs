//! Wall-clock benchmarks for `HMC` and `ManualHMC`, migrated from `#[ignore]`d tests in
//! `src/hmc.rs`. Run with `cargo bench --bench hmc_bench` (add `--features wgpu` for the
//! WGPU benchmark). These measure timing only; correctness is covered by the crate's
//! regular (non-ignored) test suite.

use burn::backend::{Autodiff, NdArray};
use criterion::{criterion_group, criterion_main, Criterion};
use mini_mcmc::core::{init, init_det, ChainRunner};
use mini_mcmc::distributions::{Rosenbrock2D, RosenbrockND};
use mini_mcmc::hmc::{ManualHMC, HMC};
use std::hint::black_box;

type BackendType = Autodiff<NdArray>;

/// Ported from `hmc::tests::test_bench_noprogress`.
fn bench_noprogress(c: &mut Criterion) {
    c.bench_function("hmc_noprogress_6chains_5000collect", |b| {
        b.iter(|| {
            let target = Rosenbrock2D {
                a: 1.0_f32,
                b: 100.0_f32,
            };
            let initial_positions = init(6, 2);
            let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
                target,
                initial_positions,
                0.01,
                50,
            )
            .set_seed(42);
            let sample = sampler.run(5000, 500);
            assert_eq!(sample.dims(), [6, 5000, 2]);
            black_box(sample);
        });
    });
}

/// Ported from `hmc::tests::test_progress_bench`.
fn bench_progress(c: &mut Criterion) {
    c.bench_function("hmc_progress_6chains_1000collect", |b| {
        b.iter(|| {
            let target = Rosenbrock2D {
                a: 1.0_f32,
                b: 100.0_f32,
            };
            let n_chains = 6;
            let initial_positions = vec![vec![1.0_f32, 2.0_f32]; n_chains];
            let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
                target,
                initial_positions,
                0.01,
                50,
            )
            .set_seed(42);
            let (sample, _stats) = sampler.run_progress(1000, 1000).unwrap();
            assert_eq!(sample.dims(), [n_chains, 1000, 2]);
            black_box(sample);
        });
    });
}

/// Ported from `hmc::tests::test_bench_10000d`.
fn bench_10000d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_10000d");
    group.sample_size(10);
    group.bench_function("run_6chains_100collect", |b| {
        b.iter(|| {
            let d = 10000;
            let n_chains = 6;
            let initial_positions: Vec<Vec<f32>> = init_det(n_chains, d);
            let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
                RosenbrockND {},
                initial_positions,
                0.01,
                50,
            )
            .set_seed(42);
            let sample = sampler.run(100, 100);
            assert_eq!(sample.dims(), [n_chains, 100, d]);
            black_box(sample);
        });
    });
    group.finish();
}

/// Ported from `hmc::tests::test_progress_10000d_bench`.
#[cfg(feature = "wgpu")]
fn bench_10000d_wgpu_progress(c: &mut Criterion) {
    use burn::backend::Wgpu;
    type WgpuBackend = Autodiff<Wgpu>;

    let mut group = c.benchmark_group("hmc_10000d_wgpu");
    group.sample_size(10);
    group.bench_function("run_progress_6chains_100collect", |b| {
        b.iter(|| {
            let d = 10000;
            let n_chains = 6;
            let initial_positions: Vec<Vec<f32>> = init_det(n_chains, d);
            let mut sampler = HMC::<f32, WgpuBackend, RosenbrockND>::new(
                RosenbrockND {},
                initial_positions,
                0.01,
                50,
            )
            .set_seed(42);
            let (sample, _stats) = sampler.run_progress(100, 100).unwrap();
            assert_eq!(sample.dims(), [n_chains, 100, d]);
            black_box(sample);
        });
    });
    group.finish();
}

/// Ported from `hmc::tests::manual_hmc_vs_hmc_bench_10000d`: compares `HMC` (burn
/// autodiff) against `ManualHMC` (analytic gradient) on the same high-dimensional
/// target, as two named benchmarks in the same group for direct comparison.
fn bench_manual_vs_burn_10000d(c: &mut Criterion) {
    let d = 10000;
    let n_chains = 6;
    let initial_positions: Vec<Vec<f32>> = init_det(n_chains, d);

    let mut group = c.benchmark_group("hmc_manual_vs_burn_10000d");
    group.sample_size(10);

    group.bench_function("burn_autodiff", |b| {
        b.iter(|| {
            let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
                RosenbrockND {},
                initial_positions.clone(),
                0.01,
                50,
            )
            .set_seed(42);
            let sample = sampler.run(100, 100);
            assert_eq!(sample.dims(), [n_chains, 100, d]);
            black_box(sample);
        });
    });

    group.bench_function("manual_analytic_gradient", |b| {
        b.iter(|| {
            let mut sampler =
                ManualHMC::new(RosenbrockND {}, initial_positions.clone(), 0.01, 50).set_seed(42);
            let sample = sampler.run(100, 100).unwrap();
            assert_eq!(sample.shape(), [n_chains, 100, d]);
            black_box(sample);
        });
    });

    group.finish();
}

#[cfg(feature = "wgpu")]
criterion_group!(
    benches,
    bench_noprogress,
    bench_progress,
    bench_10000d,
    bench_10000d_wgpu_progress,
    bench_manual_vs_burn_10000d,
);
#[cfg(not(feature = "wgpu"))]
criterion_group!(
    benches,
    bench_noprogress,
    bench_progress,
    bench_10000d,
    bench_manual_vs_burn_10000d,
);
criterion_main!(benches);
