use ndarray::LinalgScalar;
use num_traits::Float;
use rand_distr::uniform::SampleUniform;
use rand::distr::Distribution as RandDistribution;
// Bind to rand's Distribution to avoid trait mismatches from other deps pulling rand 0.8.
use rand_distr::StandardNormal;
use rand::Rng;

/// Abstraction over a mutable Euclidean vector that supports the in-place
/// operations required by the Hamiltonian integrators.
pub trait EuclideanVector: Clone {
    type Scalar: Float + LinalgScalar + SampleUniform + Copy;

    /// Returns the dimensionality of the vector.
    fn len(&self) -> usize;

    /// Creates a zero-initialized vector with the same shape.
    fn zeros_like(&self) -> Self;

    /// Resets the vector to all zeros in-place.
    fn fill_zero(&mut self);

    /// Copies the contents of `other` into `self` without reallocating.
    fn assign(&mut self, other: &Self);

    /// In-place addition.
    fn add_assign(&mut self, other: &Self);

    /// In-place fused multiply-add: `self += alpha * other`.
    fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar);

    /// Scales the vector in-place.
    fn scale_assign(&mut self, alpha: Self::Scalar);

    /// Dot product between two vectors.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// Fills the vector with samples from N(0, 1) in-place.
    fn fill_standard_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>;

    /// Writes the vector contents into the provided slice.
    fn write_to_slice(&self, out: &mut [Self::Scalar]);
}

impl<T> EuclideanVector for ndarray::Array1<T>
where
    T: Float + LinalgScalar + SampleUniform + Copy,
    StandardNormal: RandDistribution<T>,
{
    type Scalar = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn zeros_like(&self) -> Self {
        ndarray::Array1::zeros(self.len())
    }

    fn fill_zero(&mut self) {
        self.fill(T::zero());
    }

    fn assign(&mut self, other: &Self) {
        self.clone_from(other);
    }

    fn add_assign(&mut self, other: &Self) {
        ndarray::Zip::from(self).and(other).for_each(|a, b| {
            *a = *a + *b;
        });
    }

    fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar) {
        ndarray::Zip::from(self).and(other).for_each(|a, b| {
            *a = *a + *b * alpha;
        });
    }

    fn scale_assign(&mut self, alpha: Self::Scalar) {
        self.mapv_inplace(|x| x * alpha);
    }

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.dot(other)
    }

    fn fill_standard_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>,
    {
        self.iter_mut()
            .for_each(|x| *x = rng.sample(StandardNormal));
    }

    fn write_to_slice(&self, out: &mut [Self::Scalar]) {
        assert_eq!(
            out.len(),
            self.len(),
            "write_to_slice called with mismatched buffer length"
        );
        let slice = self
            .as_slice()
            .expect("Array1 is expected to be contiguous when writing to slice");
        out.copy_from_slice(slice);
    }
}

#[cfg(feature = "burn")]
mod burn_impl {
    use super::EuclideanVector;
    use burn::prelude::{Backend, Tensor, TensorData};
    use burn::tensor::Element;
    use burn::tensor::ElementConversion;
    use num_traits::{Float, FromPrimitive};
    use rand::distr::Distribution as RandDistribution;
    use rand_distr::uniform::SampleUniform;
    use rand_distr::StandardNormal;
    use rand::Rng;

    impl<T, B> EuclideanVector for Tensor<B, 1>
    where
        T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
        B: Backend<FloatElem = T>,
        StandardNormal: RandDistribution<T>,
    {
        type Scalar = T;

        fn len(&self) -> usize {
            self.dims()[0]
        }

        fn zeros_like(&self) -> Self {
            Tensor::<B, 1>::zeros_like(self)
        }

        fn fill_zero(&mut self) {
            let zeros = Tensor::<B, 1>::zeros_like(self);
            self.inplace(|_| zeros.clone());
        }

        fn assign(&mut self, other: &Self) {
            self.inplace(|_| other.clone());
        }

        fn add_assign(&mut self, other: &Self) {
            self.inplace(|x| x.add(other.clone()));
        }

        fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar) {
            self.inplace(|x| x.add(other.clone().mul_scalar(alpha)));
        }

        fn scale_assign(&mut self, alpha: Self::Scalar) {
            self.inplace(|x| x.mul_scalar(alpha));
        }

        fn dot(&self, other: &Self) -> Self::Scalar {
            self.clone().mul(other.clone()).sum().into_scalar()
        }

        fn fill_standard_normal(&mut self, rng: &mut impl Rng)
        where
            StandardNormal: RandDistribution<Self::Scalar>,
        {
            let mut data = Vec::with_capacity(self.len());
            for _ in 0..self.len() {
                data.push(rng.sample(StandardNormal));
            }
            let noise = Tensor::<B, 1>::from_data(
                TensorData::new(data, [self.len()]),
                &B::Device::default(),
            );
            self.inplace(|_| noise.clone());
        }

        fn write_to_slice(&self, out: &mut [Self::Scalar]) {
            let data = self.to_data();
            let slice = data.as_slice().expect("Tensor data expected to be dense");
            assert_eq!(
                out.len(),
                slice.len(),
                "write_to_slice called with mismatched buffer length"
            );
            out.copy_from_slice(slice);
        }
    }
}
