use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use root1d::{bisect, toms748};
use roots;

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic");
    for c in [2., 10., 20., 40., 70., 100.] {
        let f = |x| x * x - c;
        group.bench_function(
            BenchmarkId::new("bisect", c),
            |b| b.iter(|| bisect(f, 0., 100.).atol(0.).rtol(1e-10).root())
        );
        group.bench_function(
            BenchmarkId::new("toms748", c),
            |b| b.iter(|| toms748(f, 0., 100.).atol(0.).rtol(1e-10).root())
        );
        group.bench_function(
            BenchmarkId::new("roots::brent", c),
            |b| b.iter(|| {
                let mut tol = Tol { rtol: 1e-10, atol: 0., maxiter: 100 };
                roots::find_root_brent(0., 100., &f, &mut tol)
            }));
    }
}

criterion_group!(benches, bench_copy);
criterion_main!(benches);


// For the "roots" crate.
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
