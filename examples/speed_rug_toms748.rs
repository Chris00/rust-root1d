use std::error::Error;
use root1d::{toms748_mut};
use rug::{Assign, Float};

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let prec = 100;
    let mut sum = Float::with_val(prec, 0_f64);

    let mut n = 0;
    let mut neval = 0;
    let ntimes_less_iter = 10;
    let rtol = 1e-10f64;
    let a = Float::with_val(prec, 0_f64);
    let b = Float::with_val(prec, 100_f64);
    // Allocating the state out of the loop does not bring any
    // noticable speed benefit.
    // let mut w = (a.clone(), a.clone(), a.clone(), a.clone(), a.clone(),
    //              a.clone(), a.clone(), a.clone(), a.clone(), a.clone(),
    //              a.clone(), a.clone(), a.clone(), a.clone(), a.clone(),
    //              a.clone());
    // let mut r = a.clone();
    for _ in 0 .. 10_000 / ntimes_less_iter {
        for i in 2..100 {
            n += 1;
            let c = Float::with_val(prec, i as f64);
            let f = |y: &mut Float, x: &Float| {
                neval += 1;
                y.assign(x * x - &c)
            };
            let r = toms748_mut(f, &a, &b).atol(0f64).rtol(rtol)
                //.work(&mut w)
                .root()?;
            // bisect_mut(f, &a, &b).atol(0f64).rtol(1e-10f64)
            //     //.work(&mut w)
            //     .root_mut(&mut r)?;
            sum += &r;
        }
    }

    println!("Toms748<T=rug::Float> sum: {:.18} (#eval {:.2})",
             sum * ntimes_less_iter, neval as f64 / n as f64);
    println!("    {} times less iterations, prec: {}, rtol: {:e}",
             ntimes_less_iter, prec, rtol);
    Ok(())
}
