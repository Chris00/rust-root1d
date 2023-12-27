//! Compute level sets of x sin(x)
//! See https://twitter.com/walkingrandomly/status/1544615360833507329
//!     https://twitter.com/ChrisRackauckas/status/1544743542094020615
//! and [Matlab R2023a][].
//!
//! [Matlab R2023a]: https://blogs.mathworks.com/matlab/2023/05/15/from-hpc-consultancy-to-a-faster-fzero-function-in-matlab-r2023a/

use std::{error::Error, iter, time::Instant};
use root1d::{toms748, bisect};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn Error + 'static>> {
    const N: usize = 100_000;
    let levels: Vec<_> = iter::from_fn(|| Some(1.5 * random::<f64>()))
        .take(N).collect();
    let mut out = Vec::with_capacity(N);

    let now = Instant::now();
    for lv in &levels {
        out.push(bisect(|x: f64| x * x.sin() - lv, 0., 2.).root()?);
    }
    println!("bisect: {} secs", now.elapsed().as_secs_f64());

    out.clear();
        let now = Instant::now();
    for lv in &levels {
        out.push(toms748(|x: f64| x * x.sin() - lv, 0., 2.).root()?);
    }
    println!("toms748: {} secs", now.elapsed().as_secs_f64());

    Ok(())
}
