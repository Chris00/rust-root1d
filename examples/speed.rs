// Trivial code to have a rough estimate of the speed.

use std::error::Error;
use root1d::bisect;

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.; // do something with the roots to avoid dead code removal

    let mut n = 0;
    let mut neval = 0;
    for _ in 0..10_000 {
        for i in 2..100 {
            n += 1;
            // Trivial function so that it does not dominate the
            // computation time.
            let c = i as f64;
            let f = |x| {neval += 1; x * x - c};
            let r = bisect(f, 0., 100.).atol(0.).rtol(1e-10).root()?;
            sum += r;
        }
    }

    println!("Bisect<T=f64> sum: {:.12} (#eval {:.2})", sum,
             neval as f64 / n as f64);
    Ok(())
}
