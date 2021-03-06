#![allow(unused_imports)]

use std::{cmp::Ordering,
          error::Error,
          fmt::{self, Display, Debug, Formatter},
          mem::swap};

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.;

    let mut n = 0;
    let mut neval = 0;
    for _ in 0..10_000 {
        for i in 2..100 {
            n += 1;
            let c = i as f64;
            let f = |x| {neval += 1; x * x - c};
            let r = bisect(f, 0., 100., 1e-10)?;
            sum += r;
        }
    }

    println!("bisect_f64 sum: {:.12} (#eval {:.2})", sum,
             neval as f64 / n as f64);
    Ok(())
}

#[derive(Debug)]
enum BisectError {
    NoSignChange,
    NotFinite,
}

impl Display for BisectError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "The sign at the bound is not opposite")
    }
}

impl Error for BisectError {}

/// Minimal and monomorphic routine to estimate the maximum possible
/// speed.
fn bisect<F>(mut f: F, a: f64, b: f64, rtol: f64) -> Result<f64, BisectError>
where F: FnMut(f64) -> f64 {
    let fa = f(a);
    let fb = f(b);
    if fa == 0. { return Ok(a) }
    if fb == 0. { return Ok(b) }
    if !(fa * fb < 0.) { return Err(BisectError::NoSignChange) }
    let mut a = a;
    let mut b = b;
    if fa > 0. { swap(&mut a, &mut b) }
    // f(a) < 0 < f(b)
    let mut niter: usize = 100;
    while niter > 0 && (a - b).abs() > rtol * a.abs().max(b.abs()) {
        use Ordering::*;
        let mut x = 0.5 * (a + b);
        if !x.is_finite() { x = 0.5 * a + 0.5 * b }
        let fx = f(x);
        // match fx.partial_cmp(&0.) {
        //     Some(Greater) => b = x,
        //     Some(Less) => a = x,
        //     Some(Equal) => return Ok(x),
        //     None => return Err(BisectError::NotFinite)
        // }
        if fx < 0. { a = x }
        else if fx > 0. { b = x }
        else if fx.is_finite() { return Ok(x) }
        else { return Err(BisectError::NotFinite) }
        niter -= 1;
    }
    Ok(0.5 * (a + b))
}
