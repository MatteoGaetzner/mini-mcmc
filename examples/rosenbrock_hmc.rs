use burn::backend::{Autodiff, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use mini_mcmc::core::ChainRunner;
use mini_mcmc::hmc::{GradientTarget, HamiltonianSampler};
use mini_mcmc::io::save_parquet;
use num_traits::Float;
use plotters::prelude::*;
use std::error::Error;

// Define the Rosenbrock distribution (classic parameters: a = 1, b = 100)
// The unnormalized log-probability is given by:
//   log π(x, y) ∝ -[(a - x)² + b*(y - x²)²]
#[derive(Clone, Copy)]
struct Rosenbrock<T: Float> {
    a: T,
    b: T,
}

// Implement GradientTarget for Rosenbrock so that HMC can compute gradients.
// This uses Burn’s autodiff backend.
impl<T, B> GradientTarget<T, B> for Rosenbrock<T>
where
    T: Float + std::fmt::Debug + burn::tensor::Element,
    B: AutodiffBackend,
{
    fn log_prob_tensor(&self, theta: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Split the 1D tensor into two tensors, one for x and one for y.
        let split: Vec<Tensor<B, 1>> = theta.clone().split(1, 0);
        let (x, y) = (
            split.first().unwrap().to_owned(),
            split.last().unwrap().to_owned(),
        );
        // Compute the log probability:
        // log π(x, y) = -[(a - x)² + b*(y - x²)²]
        -(((-x.clone()).add_scalar(self.a)).powi(Tensor::<B, 1>::from_floats(
            [T::from(2.0).unwrap()],
            &B::Device::default(),
        )) + ((y.sub(x.clone().mul(x))).powi(Tensor::<B, 1>::from_floats(
            [T::from(2.0).unwrap()],
            &B::Device::default(),
        )))
        .mul_scalar(self.b))
    }
}

/// Renders the Rosenbrock HMC samples to an SVG file named "hmc_scatter_plot.svg".
pub fn plot_samples(samples: &[nalgebra::DMatrix<f32>]) -> Result<(), Box<dyn std::error::Error>> {
    // 1) Gather all points to determine the global min/max for the axes.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for mat in samples.iter() {
        for row in mat.row_iter() {
            xs.push(row[0]);
            ys.push(row[1]);
        }
    }
    let x_min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 2) Create an SVG drawing area (1200x900).
    let root = SVGBackend::new("hmc_scatter_plot.svg", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // 3) Build the chart with extra space around the axes.
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "HMC Samples from Rosenbrock Distribution",
            ("sans-serif", 45),
        )
        .margin(20)
        .x_label_area_size(60) // more space for x-axis
        .y_label_area_size(60) // more space for y-axis
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    // 4) Configure a minimal-style mesh.
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Dimension 0")
        .y_desc("Dimension 1")
        .axis_desc_style(("sans-serif", 30))
        .draw()?;

    // 5) Define a palette of RGBA colors with some transparency (alpha).
    //    If you have more than 6 chains, just extend or cycle these colors.
    let chain_colors = [
        RGBAColor(255, 0, 0, 0.5),   // semi-transparent red
        RGBAColor(0, 0, 255, 0.5),   // semi-transparent blue
        RGBAColor(0, 255, 0, 0.5),   // semi-transparent green
        RGBAColor(255, 0, 255, 0.5), // semi-transparent magenta
        RGBAColor(0, 255, 255, 0.5), // semi-transparent cyan
        RGBAColor(255, 255, 0, 0.5), // semi-transparent yellow
    ];

    // 6) Plot each chain with a distinct color, bigger dots, and alpha.
    for (chain_idx, mat) in samples.iter().enumerate() {
        let color = chain_colors[chain_idx % chain_colors.len()];

        chart
            .draw_series(mat.row_iter().map(move |row| {
                let (x, y) = (row[0], row[1]);
                Circle::new((x, y), 5, color.filled()) // radius=5 => bigger dots
            }))?
            .label(format!("Chain {}", chain_idx))
            .legend(move |(lx, ly)| Circle::new((lx, ly), 5, color.filled()));
    }

    // 7) Draw a legend (in the top-right corner by default).
    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .label_font(("sans-serif", 20))
        .draw()?;

    println!("Saved HMC scatter plot to hmc_scatter_plot.svg");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create the Rosenbrock target distribution with a = 1.0 and b = 100.0.
    let target = Rosenbrock { a: 1.0, b: 100.0 };

    // Define an initial position in 2D.
    let initial_pos = vec![0.123_f32, 1.23];

    // Create a Hamiltonian sampler with 4 parallel chains.
    // - step_size is set to a small value (e.g. 0.01) because the valley is narrow.
    // - n_leapfrog sets the number of leapfrog integration steps.
    let mut sampler: HamiltonianSampler<f32, Autodiff<Wgpu>, Rosenbrock<f32>> =
        HamiltonianSampler::new(target, initial_pos, 0.01, 10, 4).set_seed(42);

    // Run the sampler for 1,000 iterations, discarding the first 100 as burn-in.
    let samples = sampler.run_with_progress(200, 20);

    plot_samples(&samples)?;

    save_parquet(&samples, "/tmp/rosenbrock_hmc.parquet")?;

    Ok(())
}
