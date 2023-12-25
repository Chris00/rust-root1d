use std::error::Error;
use std::f64::consts::PI;
use root1d::{bisect, bisect_mut, toms748, BisectMutWorkspace};
use rug::{Assign, Float, float::Constant, ops::CompleteRound};

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let f1 = |x: f64| x.sin() - 0.5 * x;
    println!("f1: {} (bisect)", bisect(f1, PI / 2., PI).atol(0.).root()?);
    println!("f1: {} (toms748)", toms748(f1, PI / 2., PI).atol(0.).root()?);

    let pi = Float::with_val(63, Constant::Pi);
    let a = (&pi / 2f64).complete(63);
    let f2 = |y: &mut Float, x: &Float| {
        y.assign(-0.5f64 * x);
        *y += x.clone().sin();
    };
    let r = bisect_mut(f2, &a, &pi).rtol(1e-20).atol(0.).root()?;
    println!("f2: {} (prec: {})", r, r.prec());

    let mut w = BisectMutWorkspace::new(&Float::new(53));
    let a = Float::with_val(53, 0.);
    let b = Float::with_val(53, 10.);
    let mut r = a.clone();
    for i in 1..4 {
        let c = Float::with_val(53, i as f64);
        let f = |y: &mut Float, x: &Float| { y.assign(x * x); *y -= &c; };
        let mut b = bisect_mut(f, &a, &b).work(&mut w).maxiter_err(true);
        let e = b.root_mut(&mut r);
        println!("i = {} ⟹ root = {} (prec: {}; {:?})", i, r, r.prec(), e);
    }

    Ok(())
}
