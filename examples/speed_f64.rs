use std::{error::Error,
          fmt::{self, Display, Debug, Formatter},
          mem::swap};

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.;

    for _ in 0..10_000 {
        for i in 2..100 {
            let c = i as f64;
            let f = |x| x * x - c;
            let r = bisect(f, 0., 100., 1e-10)?;
            sum += r;
        }
    }

    println!("[f64] sum: {:.12}", sum);
    Ok(())
}

#[derive(Debug)]
enum BisectError {
    NoSignChange
}

impl Display for BisectError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "The sign at the bound is not opposite")
    }
}

impl Error for BisectError {}

/// Minimal and monomorphic routine to estimate the maximum possible
/// speed.
fn bisect<F>(f: F, a: f64, b: f64, rtol: f64) -> Result<f64, BisectError>
where F: Fn(f64) -> f64 {
    if f(a) == 0. { return Ok(a) }
    if f(b) == 0. { return Ok(b) }
    if !(f(a) * f(b) < 0.) { return Err(BisectError::NoSignChange) }
    let mut a = a;
    let mut b = b;
    if f(a) > 0. { swap(&mut a, &mut b) }
    // f(a) < 0 < f(b)
    while (a - b).abs() > rtol * a.abs().max(b.abs()) {
        let x = 0.5 * (a + b);  // not suitable for general use
        let fx = f(x);
        if fx < 0. { a = x }
        else if fx > 0. { b = x }
        else { return Ok(x) }
    }
    Ok(0.5 * (a + b))
}
