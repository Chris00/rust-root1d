use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use root1d::{bisect, toms748};

mod common;

fn bench_copy(c: &mut Criterion) {
    let pbm = common::toms748_test_problems();
    let rtol = 1e-7;

    let mut group = c.benchmark_group("pbm748");
    for (i, funs) in pbm.iter().enumerate() {
        // Use number indices so a comparative line graph is drawn.
        group.bench_function(
            BenchmarkId::new("bisect", i+1),
            |c| c.iter(|| -> Result<f64, root1d::Error<f64>> {
                let mut s = 0.;
                for (f, a, b) in funs.iter() {
                    s += bisect(f, *a, *b).atol(0.).rtol(rtol).root()?;
                }
                Ok(s)
            }));
        group.bench_function(
            BenchmarkId::new("toms748", i+1),
            |c| c.iter(|| -> Result<f64, root1d::Error<f64>> {
                let mut s = 0.;
                for (f, a, b) in funs.iter() {
                    s += toms748(f, *a, *b).atol(0.).rtol(rtol).root()?;
                }
                Ok(s)
            }));
        group.bench_function(
            BenchmarkId::new("roots::brent", i+1),
            |c| c.iter(|| -> Result<f64, roots::SearchError> {
                let mut s = 0.;
                for (f, a, b) in funs.iter() {
                    let mut tol = common::Tol { rtol, atol: 0., maxiter: 100 };
                    s += roots::find_root_brent(*a, *b, &f, &mut tol)?;
                }
                Ok(s)
            }));
    }
}

criterion_group!(benches, bench_copy);
criterion_main!(benches);
