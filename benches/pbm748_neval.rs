use std::cell::Cell;
use root1d::{bisect, toms748};
use roots;

mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pbm = common::toms748_test_problems();

    println!("Number of function evaluations");
    println!("Tol\tBisect\tToms748\tBrent");
    for tol in [1e-7, 1e-10, 1e-15, 0.] {
        let mut n_bisect = 0;
        let mut n_toms748 = 0;
        let n_brent = Cell::new(0); // `roots` only accepts Fn functions
        let stop = |&a: &f64, &b: &f64, fa: &f64, fb: &f64| -> bool {
            let u = if fa.abs() <= fb.abs() { a } else { b };
            let tole = 2. * u.abs() * 1.9073486328e-16 + tol;
            (b - a).abs() <= 2. * tole };
        for funs in pbm.iter() {
            for &(f, a, b) in funs.iter() {
                let f1 = |x| { n_bisect += 1;  f(x) };
                bisect(f1, a, b).terminate(stop).root()?;
                let f2 = |x| { n_toms748 += 1;  f(x) };
                toms748(f2, a, b).terminate(stop).root()?;
                let f3 = |x| { n_brent.set(n_brent.get() + 1); f(x) };
                let mut tol = Tol { f, tol };
                roots::find_root_brent(a, b, &f3, &mut tol)?;
            }
        }
        println!("{:e}\t{}\t{}\t{}",
                 tol, n_bisect, n_toms748, n_brent.get());
    }
    Ok(())
}

struct Tol<F>{ f: F, tol: f64 }

impl<F> roots::Convergency<f64> for Tol<F>
where F: Fn(f64) -> f64 {
    fn is_converged(&mut self, a: f64, b: f64) -> bool {
        let fa = ((self.f)(a)).abs();
        let fb =  ((self.f)(b)).abs();
        let u = if fa <= fb { a } else { b };
        let tole = 2. * u.abs() * 1.9073486328e-16 + self.tol;
        (b - a).abs() <= 2. * tole
    }
    fn is_root_found(&mut self, y: f64) -> bool {
        y == 0.
    }
    fn is_iteration_limit_reached(&mut self, iter: usize) -> bool {
        iter > 100
    }
}


// Local Variables:
// compile-command: "cargo bench neval"
// End:
