use std::error::Error;
use root1d::toms748;

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.;

    let mut n = 0;
    for _ in 0..10_000 {
        for i in 2..100 {
            let c = i as f64;
            let f = |x| {n += 1; x * x - c};
            let r = toms748(f, 0., 100.).atol(0.).rtol(1e-10).root()?;
            sum += r;
        }
    }

    println!("Toms<T=f64> sum: {:.12} (#eval {})", sum, n);
    Ok(())
}
