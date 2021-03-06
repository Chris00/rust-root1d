use std::error::Error;
use root1d::toms748;

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.;

    let mut n = 0;
    let mut neval = 0;
    for _ in 0..10_000 {
        for i in 2..100 {
            n += 1;
            let c = i as f64;
            let f = |x| {neval += 1; x * x - c};
            let r = toms748(f, 0., 100.).atol(0.).rtol(1e-10).root()?;
            sum += r;
        }
    }

    println!("Toms<T=f64> sum: {:.12} (#eval {:.2})", sum,
             neval as f64 / n as f64);
    Ok(())
}
