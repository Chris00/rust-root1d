use std::cell::Cell;
use root1d::{bisect, toms748, toms748_mut};
use roots;
use rug::{Assign, Float};

mod common;

type R<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone)]
struct Neval {
    bisect: u32,
    toms748: u32,
    toms748_mut: u32,
    brent: Cell<u32>,
}

impl Neval {
    fn zero() -> Neval {
        Neval { bisect: 0, toms748: 0, toms748_mut: 0, brent: Cell::new(0) }
    }
}

impl std::ops::AddAssign<&Neval> for Neval {
    fn add_assign(&mut self, other: &Self) {
        self.bisect += other.bisect;
        self.toms748 += other.toms748;
        self.toms748_mut += other.toms748_mut;
        self.brent.set(self.brent.get() + other.brent.get());
    }
}

fn add_niter<'a>(
    funs: impl IntoIterator<Item=&'a (fn(f64) -> f64, f64, f64)>,
    tol: f64, n: &mut Neval) -> R<f64> {
    let stop = |&a: &f64, &b: &f64, fa: &f64, fb: &f64| -> bool {
        let u = if fa.abs() <= fb.abs() { a } else { b };
        let tole = 2. * u.abs() * 1.9073486328e-16 + tol;
        b - a <= 2. * tole
        // const ATOL: f64 = 2e-12;
        // const RTOL: f64 = 8.881784197001252e-16;
        // b - a <= ATOL + RTOL * b.abs() // Numpy
    };
    let mut s = 0.;
    for &(f, a, b) in funs.into_iter() {
        let f1 = |x| { n.bisect += 1;  f(x) };
        bisect(f1, a, b).terminate(stop).root()?;
        let f2 = |x| { n.toms748 += 1;  f(x) };
        s += toms748(f2, a, b).terminate(stop).root()?;
        let f3 = |x| { n.brent.set(n.brent.get() + 1); f(x) };
        let mut tol = Tol { f, tol };
        roots::find_root_brent(a, b, &f3, &mut tol)?;
        let f4 = |y: &mut Float, x: &Float| {
            n.toms748_mut += 1;  y.assign(f(x.to_f64()));
        };
        let a1 = Float::with_val(53, a);
        let b1 = Float::with_val(53, b);
        let stop1 = |a: &Float, b: &Float, fa: &Float, fb: &Float| -> bool {
            stop(&a.to_f64(), &b.to_f64(), &fa.to_f64(), &fb.to_f64())
        };
        toms748_mut(f4, &a1, &b1).terminate(stop1).root()?;
    }
    Ok(s)
}

fn niter<'a>(funs: impl IntoIterator<Item=&'a (fn(f64) -> f64, f64, f64)>,
             tol: f64) -> R<(f64, Neval)> {
    let mut n = Neval::zero();
    let r = add_niter(funs, tol, &mut n)?;
    Ok((r, n))
}

fn main() -> R<()> {
    let pbm = common::toms748_test_problems();

    println!("Number of function evaluations");
    println!("Tol\tBisect\tToms748\tBrent\tToms748_mut");
    for tol in [1e-7, 1e-10, 1e-15, 0.] {
        let mut n = Neval::zero();
        add_niter(pbm.iter().flatten(), tol, &mut n)?;
        println!("{:e}\t{}\t{}\t{}\t{}",
                 tol, n.bisect, n.toms748, n.brent.get(), n.toms748_mut);
    }

    println!("With tol = 1e-15:");
    for (group, subs) in [
        ("G1", vec![(13, 0)]),
        ("G2", vec![(3, 2), (14, 9), (14, 29), (15, 10), (15, 25)]),
        ("G3", vec![(1, 0), (2, 1), (4, 5), (5, 0), (6, 5), (7, 1),
                    (8, 2), (9, 0), (10, 1), (11, 3), (12, 1)])
    ] {
        let mut tot = Neval::zero();
        for (id, sub) in subs {
            let (r, n) = niter(&[pbm[id-1][sub]], 1e-15)?;
            tot += &n;
            println!("  #{}\t{}\t{}\t{}\t{}\t({:.15e})",
                     id, n.bisect, n.toms748, n.brent.get(), n.toms748_mut, r);
        }
        println!("Tot {group}:\t{}\t{}\t{}\t{}",
                 tot.bisect, tot.toms748, tot.brent.get(), tot.toms748_mut);
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
// compile-command: "make -C.. neval"
// End:
