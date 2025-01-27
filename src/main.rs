mod mini_mcmc;
use mini_mcmc::{Gaussian2D, GaussianProposal, MetropolisHastings};

use plotters::prelude::*; // For plotting
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // target: 2D Gaussian with mean=[0,0], diag covariance=[1,1]
    let target = Gaussian2D {
        mean: [0.0, 0.0],
        cov: [[1.0, 0.0], [0.0, 1.0]],
    };
    let proposal = GaussianProposal::new(0.5);
    let initial_state = vec![1.0, 1.0];

    let mut mh = MetropolisHastings::new(target, proposal, initial_state);

    let iterations = 10_000;
    let mut samples = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let new_state = mh.step();
        samples.push(new_state);
    }

    println!("Generated {} samples", samples.len());

    // --- Plot the 2D samples ---
    // 1) Create a drawing area using BitMapBackend
    let root = BitMapBackend::new("samples.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 2) Build a chart with x and y ranges
    //    For a standard normal-like distribution, ±3 or ±4 is usually enough to see the bulk.
    let mut chart = ChartBuilder::on(&root)
        .caption("MCMC Samples from 2D Gaussian", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-4.0..4.0, -4.0..4.0)?;

    // 3) Draw the mesh (axes, grid, etc.)
    chart.configure_mesh().draw()?;

    // 4) Plot each sample as a small circle
    chart.draw_series(samples.iter().map(|point| {
        let x = point[0];
        let y = point[1];
        Circle::new((x, y), 3, RED.filled())
    }))?;

    println!("Saved scatter plot to samples.png");
    Ok(())
}
