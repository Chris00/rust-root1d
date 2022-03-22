// Trivial code to have a rough estimate of the speed.

use std::error::Error;
use root1d::bisect;

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.; // do something with the roots to avoid dead code removal

    for _ in 0..10_000 {
        for i in 2..100 {
            // Trivial function so that it does not domainate the
            // computation time.
            let c = i as f64;
            let f = |x| x * x - c;
            let r = bisect(f, 0., 100.).atol(0.).rtol(1e-10).root()?;
            sum += r;
        }
    }

    println!("[T=f64] sum: {:.12}", sum);
    Ok(())
}
