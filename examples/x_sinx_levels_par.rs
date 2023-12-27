//! Parallel version of `x_sinx_levels.rs`.

use std::{error::Error, iter, time::Instant};
use root1d::{bisect, toms748};
use rand::prelude::*;
use rayon::prelude::*;

fn main() -> Result<(), Box<dyn Error + 'static>> {
    const N: usize = 100_000;
    let levels: Vec<_> = iter::from_fn(|| Some(1.5 * random::<f64>()))
        .take(N).collect();
    let mut out = vec![0.0; N];

    let now = Instant::now();
    levels
        .par_iter()
        .map(|lv| bisect(|x: f64| x * x.sin() - lv, 0., 2.).root().unwrap())
        .collect_into_vec(&mut out);
    println!("bisect: {} secs", now.elapsed().as_secs_f64());

    let now = Instant::now();
    levels
        .par_iter()
        .map(|lv| toms748(|x: f64| x * x.sin() - lv, 0., 2.).root().unwrap())
        .collect_into_vec(&mut out);
    println!("toms748: {} secs", now.elapsed().as_secs_f64());

    Ok(())
}
