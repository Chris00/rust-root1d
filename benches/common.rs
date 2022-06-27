use std::f64::consts::{PI, E};

// Functions use in the "Algorithms 748" paper.

fn f1(x: f64) -> f64 { x.sin() - x / 2. }

fn f2(x: f64) -> f64 {
    let mut s = 0.;
    for i in 1..=20 {
        let i = i as f64;
        s += (2. * i - 5.).powi(2) / (x - i * i).powi(3);
    }
    -2. * s
}

fn f3<const A: i32, const B: i32>(x: f64) -> f64 {
    A as f64 * x * (B as f64 * x).exp()
}

fn f4_0<const N:i32>(x: f64) -> f64 { x.powi(N) - 0.2 }
fn f4_1<const N:i32>(x: f64) -> f64 { x.powi(N) - 1. }

fn f5(x: f64) -> f64 { x.sin() - 0.5 }

fn f6<const N:i32>(x: f64) -> f64 {
    let n = (-N) as f64;
    2. * x * n.exp() - 2. * (n*x).exp() + 1.
}

fn f7<const N:i32>(x: f64) -> f64 {
    (1. + (1. - N as f64).powi(2)) * x - (1. - N as f64 * x).powi(2)
}

fn f8<const N:i32>(x: f64) -> f64 { x * x - (1. - x).powi(N) }

fn f9<const N:i32>(x: f64) -> f64 {
    (1. + (1. - N as f64).powi(4)) * x - (1. - N as f64 * x).powi(4)
}

fn f10<const N:i32>(x: f64) -> f64 {
    ((-N) as f64 * x).exp() * (x - 1.) + x.powi(N)
}

fn f11<const N:i32>(x: f64) -> f64 {
    let n = N as f64;
    (n * x - 1.) / ((n - 1.) * x)
}

fn f12<const N:i32>(x: f64) -> f64 {
    let n1 = 1. / N as f64;
    x.powf(n1) - (N as f64).powf(n1)
}

fn f13(x: f64) -> f64 {
    if x == 0. { 0. }  else { x / (1. / (x * x)).exp() }
}

fn f14<const N:i32>(x: f64) -> f64 {
    let n20 = N as f64 / 20.;
    if x >= 0. { n20 * (x / 1.5 + x.sin() - 1.) } else { -n20 }
}

fn f15<const N:i32>(x: f64) -> f64 {
    if x > 2e-3 / (1 + N) as f64 {
        E - 1.859
    } else if x >= 0. {
        (0.5 * (N + 1) as f64 * x * 1e3).exp() - 1.859
    } else {
        -0.859
    }
}

/// Return the list of test problems in the “Algorithm 748” paper.
#[inline]
pub fn toms748_test_problems() -> [Vec<(fn(f64) -> f64, f64, f64)>; 15] {
    macro_rules! f { ($f: expr) => { $f as fn(f64) -> f64 } }
    macro_rules! vec_gen {
        ($f: ident $($n: literal)+, $a: expr, $b:expr) => {
            vec![$((f!($f::<$n>), $a, $b),)+]
        }
    }
    [vec![(f!(f1), 0.5 * PI, PI)],
     vec![(f!(f2),   1. + 1e-9,   4. - 1e-9), // n = 1
          (f!(f2),   4. + 1e-9,   9. - 1e-9),
          (f!(f2),   9. + 1e-9,  16. - 1e-9),
          (f!(f2),  16. + 1e-9,  25. - 1e-9),
          (f!(f2),  25. + 1e-9,  36. - 1e-9), // n = 5
          (f!(f2),  36. + 1e-9,  49. - 1e-9),
          (f!(f2),  49. + 1e-9,  64. - 1e-9),
          (f!(f2),  64. + 1e-9,  81. - 1e-9),
          (f!(f2),  81. + 1e-9, 100. - 1e-9),
          (f!(f2), 100. + 1e-9, 121. - 1e-9)], // n = 10
     vec![(f!(f3::< -40, -1>), -9., 31.),
          (f!(f3::<-100, -2>), -9., 31.),
          (f!(f3::<-200, -3>), -9., 31.)],
     { let mut t = vec_gen!(f4_0 4 6 8 10 12, 0., 5.);
       t.append(&mut vec_gen!(f4_1 4 6 8 10 12, 0., 5.));
       t.append(&mut vec_gen!(f4_1 8 10 12 14, -0.95, 4.05));
       t },
     vec![(f!(f5), 0., 1.5)],
     vec_gen!(f6 1 2 3 4 5 20 40 60 80 100, 0., 1.),
     vec_gen!(f7 5 10 20, 0., 1.),
     vec_gen!(f8 2 5 10 15 20, 0., 1.),
     vec_gen!(f9 1 2 4 5 8 15 20, 0., 1.),
     vec_gen!(f10 1 5 10 15 20, 0., 1.),
     vec_gen!(f11 2 5 15 20, 0.01, 1.),
     vec_gen!(f12 2 3 4 5 6 7 9 11 13 15 17 19 21 23 25 27 29 31 33,
              1., 100.),
     vec![(f!(f13), -1., 4.)],
     vec_gen!(f14 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
              22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40,
              -1e4, PI / 2.),
     vec_gen!(f15 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37
              38 39 40 100 200 300 400 500 600 700 800 900 1000,
              -1e4, 1e-4)
    ]
}



/// Tolerances for the "roots" crate.
pub struct Tol {
    pub rtol: f64,
    pub atol: f64,
    pub maxiter: usize,
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
