use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use mini_mcmc::hmc::{GradientTarget, HMC};
use num_traits::Float;
use plotters::prelude::*;
use std::{error::Error, time::Instant};

// Define the Rosenbrock distribution.
#[derive(Clone, Copy)]
struct Rosenbrock<T: Float> {
    a: T,
    b: T,
}

impl<T, B> GradientTarget<T, B> for Rosenbrock<T>
where
    T: Float + std::fmt::Debug + Element,
    B: burn::tensor::backend::AutodiffBackend,
{
    fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1> {
        let n = positions.dims()[0] as i64;
        let x = positions.clone().slice([(0, n), (0, 1)]);
        let y = positions.clone().slice([(0, n), (1, 2)]);

        // Compute (a - x)^2 in place.
        let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);

        // Compute (y - x^2)^2 in place.
        let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);

        // Return the negative sum as a flattened 1D tensor.
        -(term_1 + term_2).flatten(0, 1)
    }
}

/// Plots the 3D tensor of samples (with shape [n_steps, n_chains, 2])
/// to an SVG file named "hmc_scatter_plot.svg".
fn plot_samples_from_tensor<B>(samples: &Tensor<B, 3>) -> Result<(), Box<dyn Error>>
where
    B: burn::tensor::backend::Backend,
{
    // Get the dimensions: samples has shape [n_steps, n_chains, 2].
    let dims = samples.dims();
    let n_steps = dims[0];
    let n_chains = dims[1];
    let dim = dims[2];
    assert_eq!(dim, 2, "Expected 2D positions for plotting");

    // Convert the tensor data to a flat Vec<f32>.
    let flat: Vec<f32> = samples.to_data().to_vec::<f32>().unwrap();

    // Reconstruct per-chain points.
    let mut chains: Vec<Vec<(f32, f32)>> = vec![Vec::with_capacity(n_steps); n_chains];
    for step in 0..n_steps {
        (0..n_chains).for_each(|chain_idx| {
            let base = step * n_chains * dim + chain_idx * dim;
            let x = flat[base];
            let y = flat[base + 1];
            chains[chain_idx].push((x, y));
        });
    }

    // Compute global x and y bounds.
    let (x_min, x_max) = chains
        .iter()
        .flat_map(|points| points.iter().map(|&(x, _)| x))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), v| {
            (min.min(v), max.max(v))
        });
    let (y_min, y_max) = chains
        .iter()
        .flat_map(|points| points.iter().map(|&(_, y)| y))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), v| {
            (min.min(v), max.max(v))
        });

    // Create an SVG drawing area (1200x900).
    let root = SVGBackend::new("hmc_scatter_plot.svg", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build the chart.
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "HMC Samples from Rosenbrock Distribution",
            ("sans-serif", 45),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Dimension 0")
        .y_desc("Dimension 1")
        .axis_desc_style(("sans-serif", 30))
        .draw()?;

    // Define a palette of semi-transparent colors.
    let chain_colors = [
        RGBAColor(255, 0, 0, 0.1),   // red
        RGBAColor(0, 0, 255, 0.1),   // blue
        RGBAColor(0, 255, 0, 0.1),   // green
        RGBAColor(255, 0, 255, 0.1), // magenta
        RGBAColor(0, 255, 255, 0.1), // cyan
        RGBAColor(255, 255, 0, 0.1), // yellow
    ];

    // Plot each chain with a distinct color.
    for (chain_idx, points) in chains.iter().enumerate() {
        let color = chain_colors[chain_idx % chain_colors.len()];
        chart
            .draw_series(
                points
                    .iter()
                    .map(move |&(x, y)| Circle::new((x, y), 5, color.filled())),
            )?
            .label(format!("Chain {}", chain_idx))
            .legend(move |(lx, ly)| Circle::new((lx, ly), 5, color.filled()));
    }

    // Draw the legend.
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
    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the Rosenbrock target.
    let target = Rosenbrock {
        a: 1.0_f32,
        b: 100.0_f32,
    };

    // Define 6 chains, each initialized to (1.0, 2.0).
    let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 6];
    let n_steps = 5000;

    // Create the data-parallel HMC sampler.
    let mut sampler = HMC::<f32, BackendType, Rosenbrock<f32>>::new(
        target,
        initial_positions,
        0.01, // step size
        50,   // number of leapfrog steps per update
        42,   // RNG seed
    );

    let start = Instant::now();
    // Run HMC for n_steps, collecting samples as a 3D tensor.
    let samples = sampler.run(n_steps, 0);

    let duration = start.elapsed();
    println!(
        "HMC sampler: generating {} samples took {:?}",
        samples.dims()[0..2].iter().product::<usize>(),
        duration
    );

    // Plot the samples using our helper.
    plot_samples_from_tensor(&samples)?;

    Ok(())
}
