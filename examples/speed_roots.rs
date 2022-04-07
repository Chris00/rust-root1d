use std::error::Error;
use roots;

fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let mut sum = 0.;

    for _ in 0..10_000 {
        for i in 2..100 {
            let c = i as f64;
            let f = |x| x * x - c;
            let mut tol = Tol { rtol: 1e-10, atol: 0., maxiter: 100 };
            let r = roots::find_root_regula_falsi(0., 100., &f, &mut tol)?;
            sum += r;
        }
    }

    println!("[roots] sum: {:.12}", sum);
    Ok(())
}


struct Tol {
    rtol: f64,
    atol: f64,
    maxiter: usize,
}

impl roots::Convergency<f64> for Tol {
    fn is_root_found(&mut self, y: f64) -> bool {
        y == 0.
    }

    fn is_converged(&mut self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.rtol * a.abs().max(b.abs()) + self.atol
    }

    fn is_iteration_limit_reached(&mut self, iter: usize) -> bool {
        iter > self.maxiter
    }
}
