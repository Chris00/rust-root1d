//! One dimensional root finding algorithms.
//!
//!
//!
//! ```
//! use root1d::bisect;
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
//! assert!((bisect(|x| x*x - 2., 0., 2.).atol(0.).root()?
//!          - 2f64.sqrt()).abs() < 1e-15);
//! # Ok(()) }
//! ```
//!
//! # Use with your own types
//!
//! This library can readily be usef with types `f64` and `f32` and,
//! if you activate the feature `rug` with `rug::Float` and
//! `rug::Rational`.  To use it with with another type, say `t`,
//! implement the trait [`RootBase`] for `t` which in turn requires
//! that you decide which type will store the default termination
//! routine (for example one based on tolerances, either
//! [`Tol<t>`][Tol] or a structure implementing [`SetTolerances`]) and
//! implement [`Terminate`] and [`Default`] for it.  To use [`bisect`]
//! you must in addition implement [`Bisectable`] for `t` and, for
//! [`toms748`], you must implement the trait [``].

use std::{cmp::Ordering,
          fmt::{self, Debug, Display, Formatter},
          mem::swap,
          result::Result};

/// Errors that may be returned by the root finding methods.
#[derive(Debug)]
pub enum Error<T> {
    /// Error indicating that the function evaluated at `x` returned
    /// the non-finite value `fx`.
    NotFinite { x: T, fx: T },
    /// Report that the maximum number of iterations has been reached,
    /// when option `maxiter_err` is turned on.  The argument is the
    /// current estimate of the root at that moment.
    MaxIter,
}
impl<T: Display> Display for Error<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFinite { x, fx } => {
                write!(f, "evaluating the function at {} yields {} which is \
                           not finite", x, fx)
            }
            Error::MaxIter => {
                write!(f, "maximum number of iterations reached")
            }
        }
    }
}

impl<T: Debug + Display> std::error::Error for Error<T> {}

////////////////////////////////////////////////////////////////////////
//
// Termination

/// Trait for termination criteria of the bracketing algorithms.
pub trait Terminate<T> {
    /// Return `true` if the bracketing [`a`, `b`] of the root is
    /// deemed good enough.
    ///
    /// This function may mutate `self` as it may contain resources
    /// that are not reallocated at every call of `stop`.
    fn stop(&mut self, a: &T, b: &T) -> bool;
}

/// Indicate that the type `Self` uses relative and absolute
/// tolerances that can be updated from type `U`.
pub trait SetTolerances<U> {
    /// Set the relative tolerance.  Must panic if `rtol` is invalid.
    fn set_rtol(&mut self, rtol: U);
    /// Set the absolute tolerance.  Must panic if `atol` is invalid.
    fn set_atol(&mut self, atol: U);
}

macro_rules! impl_options {
    // Implement the options to set the termination criterion (of type
    // `Term`) held by the field `t` of the struct `$s` also having
    // the additional `field`s in addition to `maxiter` and `maxiter_err`.
    ($s: ident, $s_tr: ident, $( $field: ident ),* ) => {
        /// Set the maximum number of iterations.
        ///
        /// Panics if `n` is ≤ 0.
        pub fn maxiter(mut self, n: usize) -> Self {
            if n <= 0 {
                panic!("root1d::Root.maxiter: the argument must be > 0");
            }
            self.maxiter = n;
            self
        }

        /// If `err` is `true` report the reach of the maximum number
        /// of iterations as an error.  Otherwise, just stop working
        /// and provide the estimate of the root after the maximum
        /// number of iterations.
        pub fn maxiter_err(mut self, err: bool) -> Self {
            self.maxiter_err = err;
            self
        }

        /// Change the termination criterion to `t`.
        pub fn terminate<Tr>(self, t: Tr) -> $s_tr!(Tr)
        where Tr: Terminate<T> {
            // FIXME: type changing struct updating is experimental
            // $s { t, .. self }
            $s { t,
                 maxiter: self.maxiter,
                 maxiter_err: self.maxiter_err,
                 $( $field: self.$field, )* }
        }

        /// Set the the relative tolerance termination criterion (that
        /// implements [`SetTolerances`]), leaving unchanged the value
        /// of the absolute tolerance.
        pub fn rtol<U>(mut self, rtol: U) -> Self
        where Term: SetTolerances<U> {
            self.t.set_rtol(rtol);
            self
        }
        /// Set the the absolute tolerance termination criterion (that
        /// implements [`SetTolerances`]), leaving unchanged the value
        /// of the relative tolerance.
        pub fn atol<U>(mut self, atol: U) -> Self
        where Term: SetTolerances<U> {
            self.t.set_atol(atol);
            self
        }
    }
}


/// Termination criterion based on a relative tolerance `rtol` an
/// absolute tolerance `atol`.
///
/// A direct use of this type as a termination criterion is possible
/// for `f32` and `f64`.  To use is a such for another type `T`,
/// implement `Terminate<T>` for `Tol<U>`.  If your termination
/// criterion requires additional resources to perform its
/// computations, it is recommended that you create instead a new
/// struct embedding these resources and implement `Terminate<T>` and
/// `SetTolerances<U>` for it (and, for convenience, `From<Tol<U>>`).
#[derive(Clone)]
pub struct Tol<U> { pub rtol: U,  pub atol: U }

macro_rules! impl_traits_tol {
    ($t: ty, $rtol: expr, $atol: expr) => {
        impl Default for Tol<$t> {
            fn default() -> Self { Tol { rtol: $rtol, atol: $atol } }
        }
        impl Terminate<$t> for Tol<$t> {
            fn stop(&mut self, a: &$t, b: &$t) -> bool {
                (a - b).abs() <= self.rtol * a.abs().max(b.abs()) + self.atol
            }
        }
        // We only implement this for certain types (and not
        // generically) so that a different assignment is possible for
        // non-copy types `$t`.
        impl SetTolerances<$t> for Tol<$t> {
            fn set_rtol(&mut self, rtol: $t) {
                self.rtol = rtol;
            }
            fn set_atol(&mut self, atol: $t) {
                self.atol = atol;
            }
        }
    }
}

impl_traits_tol!(f64, 4. * f64::EPSILON, 2e-12);
impl_traits_tol!(f32, 4. * f32::EPSILON, 2e-6);

////////////////////////////////////////////////////////////////////////
//
// Base trait

/// Base trait that a type must implement to use any root finding method.
pub trait RootBase: Clone + Debug {
    /// Type for the default termination criteria.
    type DefaultTerminate: Default + Terminate<Self>;

    /// Returns `true` iff `self` is finite.
    fn is_finite(&self) -> bool;

    /// Return the neutral element for the addition.
    fn cmp0(&self) -> Option<Ordering>;

    /// Set `self` to `rhs` using if possible the ressources already
    /// allocated for `self`.
    fn assign(&mut self, rhs: &Self);
}

macro_rules! root_base_fXX {
    ($t: ty) => {
        impl RootBase for $t {
            fn is_finite(&self) -> bool { Self::is_finite(*self) }
            fn cmp0(&self) -> Option<Ordering> { self.partial_cmp(&0.0) }
            fn assign(&mut self, rhs: &Self) { *self = *rhs }
            type DefaultTerminate = Tol<$t>;
        }
    }
}

root_base_fXX!(f64);
root_base_fXX!(f32);

////////////////////////////////////////////////////////////////////////
//
// Bisectable types

/// Trait indicating that the type is suitable for the bisection algorithm.
pub trait Bisectable: RootBase {
    /// Set `self` to the midpoint of the interval \[`a`, `b`\].
    /// The bounds may be assumed to be finite (as determined by
    /// [`RootBase::is_finite`]).
    fn assign_mid(&mut self, a: &Self, b: &Self);
}

macro_rules! bisectable_fXX {
    ($t: ty) => {
        impl Bisectable for $t {
            fn assign_mid(&mut self, a: &Self, b: &Self) {
                // Based on: F. Goualard, “How do you compute the
                // midpoint of an interval?,” ACM Trans. Math. Softw.,
                // vol. 40, no. 2, pp. 1–25, Feb. 2014, doi: 10.1145/2493882.
                let mid = 0.5 * (a + b);
                if mid.is_finite() { *self = mid }
                else { *self = 0.5 * a + 0.5 * b }
            }
        }
    }
}

bisectable_fXX!(f64);
bisectable_fXX!(f32);


////////////////////////////////////////////////////////////////////////
//
// Bisection for copy types

/// Bisect the function `f` on the interval \[`a`, `b`\].  Nothing is
/// computed until the [`root`][Bisect::root] or
/// [`root_mut`][Bisect::root_mut] method is used on the result.
///
/// The default maximum number of iterations is 100 and reaching that
/// many iteration simply returns the root (you can report that as an
/// error by calling [`maxiter_err`][Bisect::maxiter]`(true)`).
///
/// This method requires that [`Bisectable`] is implemented for the
/// type `T`.  This in turn requires that [`RootBase`] is implemented
/// for `T` which provides the default termination criteria.
#[must_use]
pub fn bisect<T,F>(f: F, a: T, b: T) -> Bisect<T, F, T::DefaultTerminate>
where T: RootBase + Copy,
      F: FnMut(T) -> T {
    if !a.is_finite() {
        panic!("root1d::root: a = {:?} must be finite", a)
    }
    if !b.is_finite() {
        panic!("root1d::root: b = {:?} must be finite", b)
    }
    Bisect { f, a, b,
             t: T::DefaultTerminate::default(),
             maxiter: 100,
             maxiter_err: false,
    }
}

/// Bisection algorithm (for copy types).
pub struct Bisect<T, F, Term>
where Term: Terminate<T> {
    f: F,
    a: T,     // `a` and `b` are the bounds of the interval.
    b: T,
    t: Term,  // Termination criterion
    maxiter: usize,
    maxiter_err: bool,
}

macro_rules! bisect_tr {
    ($tr: ty) => { Bisect<T, F, $tr> }
}

impl<T, F, Term> Bisect<T, F, Term>
where T: RootBase + Copy, Term: Terminate<T> {
    impl_options!(Bisect, bisect_tr,  f, a, b);
}

impl<T, F, Term> Bisect<T, F, Term>
where T: Bisectable + Copy,
      F: FnMut(T) -> T,
      Term: Terminate<T> {
    /// Return `Ok(r)` where `r` is a root of the function or `Err`
    /// indicating that the function returned a NaN value or, if
    /// [`maxiter_err`][Bisect::maxiter_err] was turned on, that the
    /// maximum number of iterations was reached.
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut x = self.a;
        self.root_gen(&mut x)
    }

    /// Same as [`root`][Bisect::root] but store the result in `root`.
    pub fn root_mut(&mut self, root: &mut T) -> Result<(), Error<T>> {
        self.root_gen(root).and(Ok(()))
    }

    fn root_gen(&mut self, x: &mut T) -> Result<T, Error<T>> {
        use Ordering::*;
        let mut a = self.a;  // `a` and `b` finite by construction
        let mut b = self.b;
        let fa = (self.f)(a);
        let fb = (self.f)(b);
        match (fa.cmp0(), fb.cmp0()) {
            (Some(Equal), _) => return Ok(a),
            (_, Some(Equal)) => return Ok(b),
            (Some(Greater), Some(Greater)) | (Some(Less), Some(Less)) => {
                panic!("root1d::bisect: the function must have opposite \
                        signs on the bounds {:?} and {:?}.", a, b)
            }
            (None, _) => return Err(Error::NotFinite{ x: a,  fx: fa }),
            (_, None) => return Err(Error::NotFinite{ x: b,  fx: fb }),
            (Some(Greater), Some(Less)) => swap(&mut a, &mut b),
            (Some(Less), Some(Greater)) => (),
        }
        // f(a) < 0 < f(b)
        x.assign_mid(&a, &b);
        while self.maxiter >0 {
            x.assign_mid(&a, &b);
            if self.t.stop(&a, &b) {
                return Ok(*x);
            }
            let fx = (self.f)(*x);
            match fx.cmp0() {
                Some(Greater) => b = *x,
                Some(Less) => a = *x,
                Some(Equal) => return Ok(*x),
                None => return Err(Error::NotFinite{ x: *x, fx }),
            }
            self.maxiter -= 1;
        }

        if self.maxiter_err {
            Err(Error::MaxIter)
        } else {
            x.assign_mid(&a, &b);
            Ok(*x)
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
// Bisection for non-copy types

/// Bisect the function `f` on the interval \[`a`, `b`\].  Nothing is
/// computed until the [`root`][BisectMut::root] or
/// [`root_mut`][BisectMut::root_mut] method is used on the result.
///
/// The default maximum number of iterations is 100 and reaching that
/// many iteration simply returns the root (you can report that as an
/// error by calling [`maxiter_err`][BisectMut::maxiter]`(true)`).
///
/// This method requires that [`Bisectable`] is implemented for the
/// type `T`.  This in turn requires that [`RootBase`] is implemented
/// for `T` which provides the default termination criteria.
#[must_use]
pub fn bisect_mut<'a,T,F>(f: F, a: &'a T, b: &'a T)
                        -> BisectMut<'a, T, F, T::DefaultTerminate>
where T: RootBase,
      F: FnMut(&mut T, &T) + 'a {
    if !a.is_finite() {
        panic!("root1d::root_mut: a = {:?} must be finite", a)
    }
    if !b.is_finite() {
        panic!("root1d::root_mut: b = {:?} must be finite", b)
    }
    BisectMut { f,  a,  b,
                t: T::DefaultTerminate::default(),
                work: None,
                maxiter: 100,
                maxiter_err: false,
    }
}

/// Bisection algorithm (for non-copy types).
pub struct BisectMut<'a, T, F, Term>
where Term: Terminate<T> {
    f: F,
    a: &'a T,
    b: &'a T,
    t: Term,
    work: Option<&'a mut (T,T,T)>, // temp vars
    maxiter: usize,
    maxiter_err: bool,
}

macro_rules! bisect_mut_tr {
    ($tr: ty) => { BisectMut<'a, T, F, $tr> }
}

impl<'a, T, F, Term> BisectMut<'a, T, F, Term>
where T: RootBase, Term: Terminate<T> {
    impl_options!(BisectMut, bisect_mut_tr,  f, a, b, work);

    /// Provide variables that will be used as workspace when running
    /// the bisection algorithm.
    pub fn work(mut self, w: &'a mut (T, T, T)) -> Self {
        self.work = Some(w);
        self
    }
}

macro_rules! return_notfinite {
    ($x: expr, $fx: expr) => {
        // The error must own its data as it is destined to be
        // propagated up, likely outside the scope where the
        // references are valid.
        return Err(Error::NotFinite{ x: $x.clone(), fx: $fx.clone() })
    }
}

impl<'a, T, F, Term> BisectMut<'a, T, F, Term>
where T: Bisectable,
      F: FnMut(&mut T, &T),
      Term: Terminate<T> {
    /// Set `root` to a root of the function `f` (see [`bisect_mut`]).
    /// Return `Some(err)` to indicate that the function `f` returned
    /// a NaN value.
    pub fn root_mut(&mut self, root: &mut T) -> Result<(), Error<T>> {
        use Ordering::*;
        let mut tmp;
        let (a, b, fx) = match &mut self.work {
            None => {
                tmp = (self.a.clone(), self.a.clone(), self.a.clone());
                (&mut tmp.0, &mut tmp.1, &mut tmp.2)
            }
            Some(v) => {
                (&mut v.0, &mut v.1, &mut v.2)
            }
        };
        a.assign(self.a);
        b.assign(self.b);
        (self.f)(fx, a);
        let fa_cmp_z = (*fx).cmp0();
        (self.f)(fx, b);
        match fa_cmp_z {
            Some(Equal) => { root.assign(a);  return Ok(()); },
            None => return_notfinite!(a, fx),
            _ => (),
        }
        let fb_cmp_z = (*fx).cmp0();
        match fb_cmp_z {
            Some(Equal) => { root.assign(b);  return Ok(()) },
            None => return_notfinite!(b, fx),
            _ => (),
        }
        match (fa_cmp_z, fb_cmp_z) {
            (Some(Greater), Some(Greater)) | (Some(Less), Some(Less)) => {
                panic!("root1d::bisect_mut: the function must have opposite \
                        signs on the bounds {:?} and {:?}.", a, b)
            }
            (Some(Greater), Some(Less)) => {
                swap(a,b);
            }
            _ => (),
        }
        // f(a) < 0 < f(b)
        while self.maxiter > 0 {
            root.assign_mid(a, b);
            if self.t.stop(a, b) {
                return Ok(());
            }
            (self.f)(fx, root);
            match (*fx).cmp0() {
                // `swap` so that we reuse allocated memory for `x`.
                Some(Greater) => swap(b, root),
                Some(Less) => swap(a, root),
                Some(Equal) => return Ok(()),
                None => return_notfinite!(root, fx),
            }
            self.maxiter -= 1;
        }

        if self.maxiter_err {
            Err(Error::MaxIter)
        } else {
            root.assign_mid(a, b);
            Ok(())
        }
    }

    /// Return a root of the function `f` (see [`bisect_mut`]) or
    /// `Err(e)` to indicate that the function `f` returned a NaN
    /// value.
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut root = self.a.clone();
        self.root_mut(&mut root).and(Ok(root))
    }
}


////////////////////////////////////////////////////////////////////////
//
// Toms 748 for copy types


/// Implements the Algorithm 748 method of Alefeld, Potro and Shi to
/// find a zero of the function f on the interval \[a,b\], where f(a)
/// and f(b) have opposite signs.
#[must_use]
pub fn toms748<T,F>(f: F, a: T, b: T) -> Toms748<T, F, T::DefaultTerminate>
where T: RootBase + Copy,
      F: FnMut(T) -> T {
    if !a.is_finite() {
        panic!("root1d::root: a = {:?} must be finite", a)
    }
    if !b.is_finite() {
        panic!("root1d::root: b = {:?} must be finite", b)
    }
    Toms748 { f, a, b,
              t: T::DefaultTerminate::default(),
              maxiter: 100,
              maxiter_err: false,
    }
}

pub struct Toms748<T, F, Term> {
    f: F,
    a: T,     // `a` and `b` are the bounds of the interval.
    b: T,
    t: Term,  // Termination criterion
    maxiter: usize,
    maxiter_err: bool,
}


////////////////////////////////////////////////////////////////////////
//
// Toms 748 for non-copy types

pub fn toms748_mut() {

}


////////////////////////////////////////////////////////////////////////
//
// Compatibility with Rug

#[cfg(feature = "rug")]
mod rug {
    use std::{cell::RefCell,
              cmp::Ordering};
    use rug::{Float, Rational, float::Round, ops::AssignRound};
    use crate::{Bisectable, RootBase, Terminate, Tol};

    /// Termination criterion based on tolerances for Rug.
    pub struct TolRug<T> {
        rtol: T,
        atol: T,
        tmp1: RefCell<T>,
        tmp2: RefCell<T>,
    }

    macro_rules! impl_rug {
        ($t: ty, $new_t: ident, $rtol: expr, $atol: expr,
         $is_finite: ident, $cmp0: ident) => {
            impl<U> From<Tol<U>> for TolRug<$t>
            where $t: AssignRound<U, Round = Round, Ordering = Ordering> {
                fn from(t: Tol<U>) -> Self {
                    let mut rtol = $new_t!();
                    rtol.assign_round(t.rtol, Round::Nearest);
                    let mut atol = $new_t!();
                    atol.assign_round(t.atol, Round::Nearest);
                    TolRug { rtol, atol,
                             tmp1: RefCell::new($new_t!()),
                             tmp2: RefCell::new($new_t!()), }
                }
            }
            impl Default for TolRug<$t> {
                fn default() -> Self {
                    TolRug { rtol: $rtol,
                             atol: $atol,
                             tmp1: RefCell::new($new_t!()),
                             tmp2: RefCell::new($new_t!()), }
                }
            }
            impl<U> super::SetTolerances<U> for TolRug<$t>
            where $t: AssignRound<U, Round = Round, Ordering = Ordering> {
                fn set_rtol(&mut self, rtol: U) {
                    self.rtol.assign_round(rtol, Round::Nearest);
                }
                fn set_atol(&mut self, atol: U) {
                    self.atol.assign_round(atol, Round::Nearest);
                }
            }
            impl Terminate<$t> for TolRug<$t> {
                fn stop(&mut self, a: &$t, b: &$t) -> bool {
                    let tmp1 = self.tmp1.get_mut();
                    let tmp2 = self.tmp2.get_mut();
                    tmp2.assign(a);
                    tmp2.abs_mut();
                    tmp1.assign(b);
                    tmp1.abs_mut();
                    if tmp1 > tmp2 {
                        tmp2.assign(&tmp1);
                    }
                    *tmp2 *= &self.rtol; // rtol * max{|a|, |b|}
                    *tmp2 += &self.atol;
                    tmp1.assign(a);
                    *tmp1 -= b;
                    tmp1.abs_mut(); // |a - b|
                    tmp1 <= tmp2
                }
            }

            impl RootBase for $t {
                type DefaultTerminate = TolRug<Self>;
                fn is_finite(&self) -> bool { $is_finite!(self) }
                fn cmp0(&self) -> Option<Ordering> { $cmp0!(self) }
                fn assign(&mut self, rhs: &Self) {
                    <Self as rug::Assign<&Self>>::assign(self, rhs); }
            }
        }
    }

    macro_rules! float_new { () => { Float::new(53) } }
    macro_rules! float_is_finite { ($s: expr) => { Float::is_finite($s) } }
    macro_rules! float_cmp0 { ($s: expr) => { $s.cmp0() } }
    impl_rug!(Float, float_new,
              Float::with_val(53, 1e-16),
              Float::with_val(53, 1e-16),
              float_is_finite, float_cmp0);

    macro_rules! rational_new { () => { Rational::new() } }
    macro_rules! rational_is_finite { ($s: ident) => { true } }
    macro_rules! rational_cmp0 { ($s: expr) => { Some($s.cmp0()) } }
    impl_rug!(Rational, rational_new,
              (1, 1000_0000_0000_0000u64).into(),
              (1, 1000_0000_0000_0000u64).into(),
              rational_is_finite, rational_cmp0);

    impl Bisectable for Float {
        #[inline]
        fn assign_mid(&mut self, a: &Self, b: &Self) {
            let p = a.prec().max(b.prec());
            self.set_prec(p + 1);
            self.assign(a);
            *self += b;
            *self /= 2i8;
        }
    }

    impl Bisectable for Rational {
        #[inline]
        fn assign_mid(&mut self, a: &Self, b: &Self) {
            self.assign(a);
            *self += b;
            *self /= 2i8;
        }
    }
}




#[cfg(all(test, feature = "rug"))]
mod tests {
    use super::bisect_mut;
    use rug::{Assign, Float};

    #[test]
    fn bisection() {
        let f = |y: &mut Float, x: &Float| {
            y.assign(x*x);
            *y -= 2u8;
        };
        let a = Float::with_val(53, 0f64);
        let b = Float::with_val(53, 2f64);
        assert!((bisect_mut(f, &a, &b).root().unwrap()
                 - 2f64.sqrt()).abs() < 1e-15);
    }
}
