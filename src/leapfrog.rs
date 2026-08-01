//! Shared plumbing for the burn-free (manual-gradient) HMC and NUTS samplers.
//!
//! Both `hmc::ManualHMC` and `nuts::ManualNUTS` advance their chains with the same
//! leapfrog integrator, operating on plain slices instead of `burn` tensors. This
//! module holds that shared integrator plus the small vector-arithmetic helpers it
//! needs, so the two samplers don't each carry their own copy.

use crate::distributions::ManualGradientTarget;
use num_traits::Float;
use rand::Rng;
use rand_distr::StandardNormal;

/// Dot product of two equal-length slices.
pub(crate) fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// In-place `y += alpha * x`.
pub(crate) fn axpy<T: Float>(y: &mut [T], alpha: T, x: &[T]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(yi, &xi)| *yi = *yi + alpha * xi);
}

/// Fills `out` with i.i.d. draws from a standard normal distribution.
pub(crate) fn fill_standard_normal<T>(out: &mut [T], rng: &mut impl Rng)
where
    T: Float,
    StandardNormal: rand_distr::Distribution<T>,
{
    out.iter_mut().for_each(|x| *x = rng.sample(StandardNormal));
}

/// Performs one leapfrog step in-place: a half-step momentum update using the
/// current gradient, a full-step position update, a gradient recomputation at the
/// new position, and a final half-step momentum update using the new gradient.
///
/// `grad` holds the gradient at `position` on entry and the gradient at the updated
/// `position` on return, so it can be reused as-is for the next call instead of being
/// reallocated. Returns the log density at the updated position.
pub(crate) fn leapfrog<T, GTarget>(
    target: &GTarget,
    position: &mut [T],
    momentum: &mut [T],
    grad: &mut [T],
    step_size: T,
) -> T
where
    T: Float,
    GTarget: ManualGradientTarget<T>,
{
    let half_step = T::from(0.5).unwrap() * step_size;
    axpy(momentum, half_step, grad);
    axpy(position, step_size, momentum);
    let logp = target.unnorm_logp_and_grad_into(position, grad);
    axpy(momentum, half_step, grad);
    logp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_computes_inner_product() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn axpy_scales_and_accumulates() {
        let mut y = [1.0, 1.0, 1.0];
        axpy(&mut y, 2.0, &[1.0, 2.0, 3.0]);
        assert_eq!(y, [3.0, 5.0, 7.0]);
    }

    /// A 1D standard-normal target: unnorm_logp(x) = -0.5*x^2, grad = -x.
    /// Leapfrog on a quadratic potential should trace a known ellipse in phase space,
    /// so this checks the integrator's arithmetic directly against a hand-derived step.
    struct StdNormal1D;

    impl ManualGradientTarget<f64> for StdNormal1D {
        fn unnorm_logp_and_grad_into(&self, position: &[f64], grad: &mut [f64]) -> f64 {
            grad[0] = -position[0];
            -0.5 * position[0] * position[0]
        }
    }

    #[test]
    fn leapfrog_matches_hand_computed_step() {
        let target = StdNormal1D;
        let mut position = [0.0];
        let mut momentum = [1.0];
        let mut grad = [0.0]; // gradient at the initial position (0.0) is 0.0
        let step_size = 0.1;

        let logp = leapfrog(&target, &mut position, &mut momentum, &mut grad, step_size);

        // mom_half = 1.0 + 0.05 * 0.0 = 1.0
        // pos'     = 0.0 + 0.1 * 1.0  = 0.1
        // grad'    = -0.1
        // mom'     = 1.0 + 0.05 * -0.1 = 0.995
        assert!((position[0] - 0.1).abs() < 1e-12);
        assert!((momentum[0] - 0.995).abs() < 1e-12);
        assert!((grad[0] - (-0.1)).abs() < 1e-12);
        assert!((logp - (-0.5 * 0.1 * 0.1)).abs() < 1e-12);
    }
}
