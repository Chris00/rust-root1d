#![allow(unused_imports)]
// Trivial code to have a rough estimate of the speed.

use std::error::Error;
use root1d::{bisect, bisect_mut};
use rug::{Assign, Float, ops::CompleteRound};

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let prec = 100;
    let mut sum = Float::with_val(prec, 0_f64);

    let n = 10;
    let rtol = 1e-11f64;
    let a = Float::with_val(prec, 0_f64);
    let b = Float::with_val(prec, 100_f64);
    // Allocating the state out of the loop does not bring any
    // noticable speed benefit.
    // let mut w = (a.clone(), a.clone(), a.clone());
    // let mut r = a.clone();
    for _ in 0 .. 10_000 / n {
        for i in 2..100 {
            let c = Float::with_val(prec, i as f64);
            let f = |y: &mut Float, x: &Float| y.assign(x * x - &c);
            let r = bisect_mut(f, &a, &b).atol(0f64).rtol(rtol)
                //.work(&mut w)
                .root()?;
            // bisect_mut(f, &a, &b).atol(0f64).rtol(1e-10f64)
            //     //.work(&mut w)
            //     .root_mut(&mut r)?;
            sum += &r;
        }
    }

    println!("Bisect<T=rug::Float> sum: {:.18}", sum * n);
    println!("    {} times less iterations, prec: {}, rtol: {:e}",
             n, prec, rtol);
    Ok(())
}
