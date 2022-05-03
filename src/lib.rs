//! One dimensional root finding algorithms.
//!
//! This crate offers several *generic* root finding algorithms for
//! functions from ℝ to ℝ.  It focuses on *bracketing* algorithms,
//! that is algorithms which start with an interval \[a,b\] such that
//! the function has opposite signs at a and b (thus containing a root
//! of the function if it is continuous) and compute a smaller
//! interval with the same property.
//!
//! # Example
//!
//! ```
//! use root1d::toms748;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! assert!((toms748(|x| x*x - 2., 0., 2.).root()?
//!          - 2f64.sqrt()).abs() < 1e-12);
//! # Ok(()) }
//! ```
//!
//! # Use with your own types
//!
//! This library can readily be used with types `f64` and `f32` and,
//! if you activate the feature `rug`, with `rug::Float` and
//! `rug::Rational`.
//!
//! To use it with with another type, say `t`,
//! implement the trait [`Bisectable`] for `t` which in turn requires
//! that you decide which type will store the default termination
//! routine (for example one based on tolerances, either
//! [`Tol<t>`][Tol] or a structure implementing [`SetTolerances`]) and
//! implement [`Terminate`] and [`Default`] for it.  To use
//! [`toms748`], you must also implement the trait [`OrdField`] or
//! [`OrdFieldMut`].

use std::{fmt::{self, Debug, Display, Formatter},
          mem::swap,
          ops::{Neg, Add, Sub, Mul, Div,
                AddAssign, SubAssign, MulAssign, DivAssign},
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
    /// deemed good enough.  `fa` and `fb` are the values of the
    /// function `f` at `a` and `b` respectively.  You can assume
    /// that `a <= b`.
    ///
    /// This function may mutate `self` as it may contain resources
    /// that are not reallocated at every call of `stop`.
    fn stop(&mut self, a: &T, b: &T, fa: &T, fb: &T) -> bool;
}

/// Indicate that the type `Self` uses relative and absolute
/// tolerances that can be updated from type `U`.
pub trait SetTolerances<U> {
    /// Set the relative tolerance.  Must panic if `rtol` is ≤ 0.
    fn set_rtol(&mut self, rtol: U);
    /// Set the absolute tolerance.  Must panic if `atol` is < 0.
    fn set_atol(&mut self, atol: U);
}

/// Enable using a closure `stop` as a termination criterion.  The
/// call `stop(a, b, fa, fb)` must say whether the interval
/// \[`a`, `b`\] is satisfactory (in which case it must return `true`)
/// or not.  `fa` and `fb` are the values of the fonction to which the
/// root-finding algorithm is applied at the bounds `a` and `b`
/// respectively.
impl<T,F> Terminate<T> for F
where F: FnMut(&T, &T, &T, &T) -> bool,
      T: Bisectable {
    fn stop(&mut self, a: &T, b: &T, fa: &T, fb: &T) -> bool {
        self(a, b, fa, fb)
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
pub struct Tol<U> {
    /// Relative tolerance.
    pub rtol: U,
    /// Absolute tolerance.
    pub atol: U }

macro_rules! impl_traits_tol_fXX {
    ($t: ty, $rtol: expr, $atol: expr) => {
        impl Default for Tol<$t> {
            fn default() -> Self { Tol { rtol: $rtol, atol: $atol } }
        }
        impl Terminate<$t> for Tol<$t> {
            #[inline]
            fn stop(&mut self, a: &$t, b: &$t, _fa: &$t, _fb: &$t) -> bool {
                b - a <= self.rtol * a.abs().max(b.abs()) + self.atol
            }
        }
        // We only implement this for certain types (and not
        // generically) so that a different assignment is possible for
        // non-copy types `$t`.
        impl SetTolerances<$t> for Tol<$t> {
            fn set_rtol(&mut self, rtol: $t) {
                if rtol <= 0. { panic!("root1d<{}>: rtol = {:e} ≤ 0",
                                      stringify!($t), rtol) }
                self.rtol = rtol;
            }
            fn set_atol(&mut self, atol: $t) {
                if atol < 0. { panic!("root1d<{}>: atol = {:e} < 0",
                                      stringify!($t), atol) }
                self.atol = atol;
            }
        }
    }
}

impl_traits_tol_fXX!(f64, 4. * f64::EPSILON, 2e-12);
impl_traits_tol_fXX!(f32, 4. * f32::EPSILON, 2e-6);


////////////////////////////////////////////////////////////////////////
//
// Bisectable types

/// Trait indicating that the type is suitable for the bisection algorithm.
pub trait Bisectable: PartialOrd + Clone + Debug {
    /// Type for the default termination criteria.
    type DefaultTerminate: Default + Terminate<Self>;

    /// Return `true` if `self` is `< 0` (thus not a NaN).
    fn lt0(&self) -> bool;

    /// Return `true` if `self` is `> 0` (thus not a NaN).
    fn gt0(&self) -> bool;

    /// Returns `true` iff `self` is finite.
    ///
    /// If `self.lt0()` and `self.gt0()` are false and
    /// `self.is_finite()` is true, then `self` must be zero.
    fn is_finite(&self) -> bool;

    /// Set `self` to `rhs` using if possible the ressources already
    /// allocated for `self`.
    fn assign(&mut self, rhs: &Self);

    /// Set `self` to the midpoint of the interval \[`a`, `b`\].
    /// The bounds may be assumed to be finite (as determined by
    /// [`Bisectable::is_finite`]).
    fn assign_mid(&mut self, a: &Self, b: &Self);
}

macro_rules! impl_bisectable_fXX {
    ($t: ty) => {
        impl Bisectable for $t {
            type DefaultTerminate = Tol<$t>;
            #[inline]
            fn lt0(&self) -> bool { *self < 0. }
            #[inline]
            fn gt0(&self) -> bool { *self > 0. }
            #[inline]
            fn is_finite(&self) -> bool { Self::is_finite(*self) }
            #[inline]
            fn assign(&mut self, rhs: &Self) { *self = *rhs }
            #[inline]
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

impl_bisectable_fXX!(f64);
impl_bisectable_fXX!(f32);


////////////////////////////////////////////////////////////////////////
//
// Common structure definition for root-finding methods

/// Define the structure to hold the root-finding options and the
/// standard methods to set these. `$l` is an optional lifetime
/// (needed for non-Copy types).
macro_rules! new_root_finding_method {
    // Function to initialize the struct.
    ($(#[$docfn: meta])* $fun: ident,
     // The structure to hold the options (and other fields).
     $(#[$doc: meta])* $struct: ident <$($l: lifetime,)? ...>,
     $($field: ident, $t: ty)*) => {
        $(#[$docfn])*
        #[must_use]
        pub fn $fun<$($l,)? T,F>(f: F, a: $(&$l)? T, b: $(&$l)? T)
                                 -> $struct<$($l,)? T, F, T::DefaultTerminate>
        where T: Bisectable {
            if !a.is_finite() {
                panic!("root1d::{}: a = {:?} must be finite",
                       stringify!($fun), a)
            }
            if !b.is_finite() {
                panic!("root1d::{}: b = {:?} must be finite",
                       stringify!($fun), b)
            }
            $struct { f,  a,  b,
                      t: T::DefaultTerminate::default(),
                      maxiter: 100,
                      maxiter_err: false,
                      $($field: None,)*  // All extra fields are options
            }
        }

        $(#[$doc])*
        pub struct $struct<$($l,)? T, F, Term>
        where Term: Terminate<T> {
            f: F,
            a: $(&$l)? T,  // `a` and `b` are the bounds of the interval.
            b: $(&$l)? T,
            t: Term,  // Termination criterion
            maxiter: usize,
            maxiter_err: bool,
            $($field: $t)*
        }

        impl<$($l,)? T, F, Term> $struct<$($l,)? T, F, Term>
        where Term: Terminate<T> {
            /// Set the maximum number of iterations.
            ///
            /// If `n`, it is interpreted as “unlimited” (actually
            /// [`usize::MAX`]).
            pub fn maxiter(mut self, n: usize) -> Self {
                if n == 0 {
                    self.maxiter = usize::MAX;
                } else {
                    self.maxiter = n;
                }
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
            ///
            /// You can use a closure `FnMut(&T, &T) -> bool` as the
            /// termination criterion `t`.
            pub fn terminate<Tr>(self, t: Tr) -> $struct<$($l,)? T, F, Tr>
            where Tr: Terminate<T> {
                // FIXME: type changing struct updating is experimental
                // $s { t, .. self }
                $struct { t,
                          f: self.f,  a: self.a,  b: self.b,
                          maxiter: self.maxiter,
                          maxiter_err: self.maxiter_err,
                          $( $field: self.$field, )* }
            }

            /// Set the the relative tolerance termination criterion (that
            /// implements [`SetTolerances`]), leaving unchanged the value
            /// of the absolute tolerance.
            ///
            /// Panics if `rtol` is ≤ 0.
            pub fn rtol<U>(mut self, rtol: U) -> Self
            where Term: SetTolerances<U> {
                self.t.set_rtol(rtol);
                self
            }
            /// Set the the absolute tolerance termination criterion (that
            /// implements [`SetTolerances`]), leaving unchanged the value
            /// of the relative tolerance.
            ///
            /// Panics if `atol` is < 0.
            pub fn atol<U>(mut self, atol: U) -> Self
            where Term: SetTolerances<U> {
                self.t.set_atol(atol);
                self
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Bisection for copy types

new_root_finding_method!(
    /// Find a root of the function `f` on the interval \[`a`, `b`\] where
    /// `f(a)` and `f(b)` have opposite signs using the bisection
    /// algorithm.
    ///
    /// The default stopping criterion for [`f64`] (resp. [`f32`]) is
    /// given by [`Tol`] with `rtol: 4. * f64::EPSILON`, and
    /// `atol: 2e-12` (resp. `rtol:4. * f32::EPSILON` and
    /// `atol: 2e-6`).  The [`Terminate`] stopping criterion is
    /// |a - b| ≤ `rtol` · max{|a|, |b|} + `atol`.
    ///
    /// The default maximum number of iterations is `100` and reaching that
    /// many iteration simply returns the root (you can report that as an
    /// error by calling [`maxiter_err`][Bisect::maxiter]`(true)`).
    /// Nothing is computed until the [`root`][Bisect::root] or
    /// [`root_mut`][Bisect::root_mut] method is used on the result.
    /// See [`Bisect`]'s methods for more options.
    ///
    /// The bisection algorithm is quite slow be requires only a few
    /// things from the type `T`.  Specifically, it requires that
    /// [`Bisectable`] is implemented for the type `T` (which also
    /// provides the default termination criteria).
    ///
    /// # Example
    ///
    /// ```
    /// use root1d::bisect;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    /// assert!((bisect(|x| x*x - 2., 0., 2.).atol(0.).root()?
    ///          - 2f64.sqrt()).abs() < 1e-15);
    /// # Ok(()) }
    /// ```
    bisect,
    /// Bisection algorithm (for [`Copy`] types).
    Bisect<...>,);

/// Indicate whether the values `fa` and `fb` of the function at
/// endpoints of the interval \[`a`, `b`\] have opposite signs or if
/// one of them vanishes.  Panic otherwise.
enum SignChange {
    /// fa < 0 < fb
    NegPos,
    /// fa > 0 > fb
    PosNeg,
    /// fa = 0
    Root1,
    /// fb = 0
    Root2,
}

/// Says whether `fa` and `fb` have opposite signs (return `NegPos` of
/// `PosNeg` in this case) or one of them is zero (return `Root1` or
/// `Root2` in this case).  Panics if it is not the case.  Return an
/// error if `fa` or `fb` is not finite.
#[inline]
fn check_sign<T>(name: &str, a: T, b: T, fa: T, fb: T)
                 -> Result<SignChange, Error<T>>
where T: Bisectable {
    use SignChange::*;
    if fa.lt0() {
        if fb.gt0() {
            Ok(NegPos)
        } else if fb.lt0() {
            panic!("{}: no change of sign, f({:?}) < 0 and f({:?}) < 0.",
                   name, a, b)
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            Err(Error::NotFinite{ x: b,  fx: fb })
        }
    } else if fa.gt0() {
        if fb.lt0() {
            Ok(PosNeg)
        } else if fb.gt0() {
            panic!("{}: no change of sign, f({:?}) > 0 and f({:?}) > 0.",
                   name, a, b)
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            Err(Error::NotFinite{ x: b,  fx: fb })
        }
    } else if fa.is_finite() { // f(a) = 0
        Ok(Root1)
    } else {
        Err(Error::NotFinite{ x: a,  fx: fa })
    }
}

impl<T, F, Term> Bisect<T, F, Term>
where T: Bisectable + Copy,
      F: FnMut(T) -> T,
      Term: Terminate<T> {
    /// Return `Ok(r)` where `r` is an approximate root of the
    /// function (provided that it is continuous) or `Err` indicating
    /// that the function returned a NaN value or, if
    /// [`maxiter_err`][Bisect::maxiter_err] was turned on, that the
    /// maximum number of iterations was reached.
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut x = self.a;
        self.root_mut(&mut x).and(Ok(x))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][Bisect::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(T,T), Error<T>> {
        let mut x = self.a;
        self.root_mut(&mut x)
    }

    /// Use the bisection algorithm to approximate a root of the
    /// function `f` on the interval \[`a`, `b`\] (see [`bisect`]).
    /// Store this approximation in `root` and return an interval
    /// \[a,b\] such that `f`(a) * `f`(b) ≤ 0 (containing `root`) and
    /// satisfying the termination criterion [`Terminate`].
    ///
    /// Note that the above description assumes that `f` is
    /// continuous.  If it is not, the description of the returned
    /// interval still holds but it is not guaranteed that `f`
    /// possesses a root in it.
    #[must_use]
    pub fn root_mut(&mut self, root: &mut T) -> Result<(T,T), Error<T>> {
        let mut a;
        let mut b;
        if self.a <= self.b {
            a = self.a;
            b = self.b;
        } else {
            a = self.b;
            b = self.a;
        };
        // `a <= b`, both finite by construction
        let mut fa = (self.f)(a);
        let mut fb = (self.f)(b);
        macro_rules! body {
            ($lt0: ident, $gt0: ident) => {
                for _ in 0 .. self.maxiter {
                    root.assign_mid(&a, &b);
                    if self.t.stop(&a, &b, &fa, &fb) {
                        return Ok((a, b));
                    }
                    let fx = (self.f)(*root);
                    if fx.$lt0() { a = *root;  fa = fx; }
                    else if fx.$gt0() { b = *root;  fb = fx; }
                    else if fx.is_finite() { return Ok((a, b)) }
                    else { return Err(Error::NotFinite{ x: *root, fx }) }
                }
            }
        }
        match check_sign("root1d::bisect", a, b, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0) } // f(a) < 0 < f(b)
            SignChange::PosNeg => { body!(gt0, lt0) },
            SignChange::Root1 => { *root = a;  return Ok((a, a))}
            SignChange::Root2 => { *root = b;  return Ok((b, b))}
        }

        if self.maxiter_err {
            Err(Error::MaxIter)
        } else {
            root.assign_mid(&a, &b);
            Ok((a, b))
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
// Bisection for non-copy types

new_root_finding_method! (
    /// Same as [`bisect`] for non-[`Copy`] types.
    ///
    /// The default maximum number of iterations is 100 and reaching that
    /// many iteration simply returns the root (you can report that as an
    /// error by calling [`maxiter_err`][BisectMut::maxiter]`(true)`).
    /// Nothing is computed until the [`root`][BisectMut::root] or
    /// [`root_mut`][BisectMut::root_mut] method is used on the result.
    /// See [`BisectMut`]'s methods for more options.
    ///
    /// This method requires that [`Bisectable`] is implemented for the
    /// type `T` which provides the default termination criteria.
    ///
    /// # Example
    ///
    /// ```
    /// use root1d::bisect_mut;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    /// let f = |y: &mut f64, x: &f64| *y = *x * *x - 2.;
    /// assert!((bisect_mut(f, &0., &2.).atol(0.).root()?
    ///          - 2f64.sqrt()).abs() < 1e-15);
    /// # Ok(()) }
    /// ```
    bisect_mut,
    /// Bisection algorithm (for non-[`Copy`] types).
    BisectMut<'a,...>,
    workspace, Option<&'a mut (T,T,T,T,T)>);

impl<'a, T, F, Term> BisectMut<'a, T, F, Term>
where T: Bisectable, Term: Terminate<T> {
    /// Provide variables that will be used as workspace when running
    /// the bisection algorithm.
    #[must_use]
    pub fn work(mut self, w: &'a mut (T,T,T,T,T)) -> Self {
        self.workspace = Some(w);
        self
    }
}

/// Same as [`check_sign`] for non-Copy types.  In addition evaluate
/// `f` at `a` and `b` and store the result in `fa` and `fb` respectively.
#[inline]
fn check_sign_mut<T,F>(name: &str, a: &T, b: &T,
                       f: &mut F, fa: &mut T, fb: &mut T)
                       -> Result<SignChange, Error<T>>
where T: Bisectable,
      F: FnMut(&mut T, &T) {
    use SignChange::*;
    f(fa, a);
    if fa.lt0() {
        f(fb, b);
        if fb.gt0() {
            Ok(NegPos)
        } else if fb.lt0() {
            panic!("{}: no change of sign, f({:?}) < 0 and f({:?}) < 0.",
                   name, a, b)
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            return Err(Error::NotFinite{ x: b.clone(), fx: fb.clone() })
        }
    } else if fa.gt0() {
        f(fb, b);
        if fb.lt0() {
            Ok(PosNeg)
        } else if fb.gt0() {
            panic!("{}: no change of sign, f({:?}) > 0 and f({:?}) > 0.",
                   name, a, b)
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            return Err(Error::NotFinite{ x: b.clone(), fx: fb.clone() })
        }
    } else if fa.is_finite() { // f(a) = 0
        Ok(Root1)
    } else {
        return Err(Error::NotFinite{ x: a.clone(), fx: fa.clone() })
    }
}

impl<'a, T, F, Term> BisectMut<'a, T, F, Term>
where T: Bisectable,
      F: FnMut(&mut T, &T),
      Term: Terminate<T> {
    /// Set `root` to a root of the function `f` (see [`bisect_mut`]).
    /// Return `Some(err)` to indicate that the algorithm failed
    /// (e.g., when the function `f` returned a NaN value).
    ///
    /// If the [`work`][`BisectMut::work`] method was not used,
    /// internal variables are constructed by cloning `root`, thereby
    /// inheriting its precision for example.
    #[must_use]
    pub fn root_mut(&mut self, root: &mut T) -> Result<(), Error<T>> {
        let mut tmp;
        let (a, b, fa, fb, fx) = match &mut self.workspace {
            None => {
                tmp = (root.clone(), root.clone(), root.clone(),
                       root.clone(), root.clone());
                (&mut tmp.0, &mut tmp.1, &mut tmp.2, &mut tmp.3, &mut tmp.4)
            }
            Some(v) =>
                (&mut v.0, &mut v.1, &mut v.2, &mut v.3, &mut v.4)
        };
        if self.a <= self.b {
            a.assign(self.a);
            b.assign(self.b);
        } else {
            a.assign(self.b);
            b.assign(self.a);
        }
        // `a <= b`, both finite by construction
        macro_rules! body {
            ($lt0: ident, $gt0: ident) => {
                for _ in 0 .. self.maxiter {
                    root.assign_mid(a, b);
                    if self.t.stop(a, b, fa, fb) {
                        return Ok(());
                    }
                    (self.f)(fx, root);
                    // `swap` so as to reuse allocated memory.
                    if fx.$lt0() { swap(a, root);
                                   swap(fa, fx); }
                    else if fx.$gt0() { swap(b, root);
                                       swap(fb, fx); }
                    else if fx.is_finite() { return Ok(()) }
                    else {
                        return Err(Error::NotFinite{ x: root.clone(),
                                                     fx: fx.clone() })
                    }
                }
            }
        }
        match check_sign_mut("root1d::bisect_mut", a, b,
                             &mut self.f, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0) } // f(a) < 0 < f(b)
            SignChange::PosNeg => { body!(gt0, lt0) }
            SignChange::Root1 => {
                root.assign(self.a);
                return Ok(())
            }
            SignChange::Root2 => {
                root.assign(self.b);
                return Ok(())
            }
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
    #[must_use]
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut root = self.a.clone();
        self.root_mut(&mut root).and(Ok(root))
    }
}


////////////////////////////////////////////////////////////////////////
//
// Toms 748 for copy types

/// Requirements on the type `T` to be able to use [`toms748`]
/// algorithm.
pub trait OrdField: Bisectable + Copy
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self> {
        /// Return `true` if `self` is the number 0.
        fn is_zero(self) -> bool;

        /// Return twice the value of `self`.
        fn twice(self) -> Self;

        /// Return the value of `self` divided by 64.
        fn div64(self) -> Self;
    }

macro_rules! impl_ordfield_fXX {
    ($t: ty) => {
        impl OrdField for $t {
            #[inline]
            fn is_zero(self) -> bool { self == 0. }
            #[inline]
            fn twice(self) -> Self { 2. * self }
            #[inline]
            fn div64(self) -> Self { 0.015625 * self }
        }
    }
}

impl_ordfield_fXX!(f32);
impl_ordfield_fXX!(f64);

new_root_finding_method!(
    /// Find a root of the function `f` on the interval \[`a`, `b`\],
    /// where `f(a)` and `f(b)` have opposite signs using Algorithm 748 by
    /// Alefeld, Potro and Shi.
    ///
    /// The default stopping criterion for [`f64`] (resp. [`f32`]) is
    /// given by [`Tol`] with `rtol: 4. * f64::EPSILON`, and
    /// `atol: 2e-12` (resp. `rtol:4. * f32::EPSILON` and
    /// `atol: 2e-6`).  The [`Terminate`] stopping criterion is
    /// |a - b| ≤ `rtol` · max{|a|, |b|} + `atol`.
    ///
    /// The default maximum number of iterations is `100` and reaching
    /// that many iteration simply returns the root (you can report
    /// that as an error with the option
    /// [`maxiter_err`][Bisect::maxiter]`(true)`).  The maximum number
    /// of iterations can be changed using the
    /// [`maxiter`][Toms748::maxiter] method.  See the methods of
    /// [`Toms748`] for more options.
    ///
    /// # Example
    ///
    /// ```
    /// use root1d::toms748;
    /// # fn main() -> Result<(), root1d::Error<f64>> {
    /// let f = |x| x * x - 2.;
    /// assert!((toms748(f, 0., 2.).atol(0.).rtol(1e-10).root()?
    ///          - 2f64.sqrt()).abs() < 1e-15);
    /// # Ok(()) }
    /// ```
    ///
    /// # Reference
    ///
    /// G. E. Alefeld, F. A. Potra, and Y. Shi, “Algorithm 748:
    /// enclosing zeros of continuous functions,” ACM Trans. Math. Softw.,
    /// vol. 21, no. 3, pp. 327–344, Sep. 1995, doi: 10.1145/210089.210111.
    toms748,
    /// [`toms748`] algorithm (for [`Copy`] types).
    Toms748<...>, );

macro_rules! bracket_sign {
    // Assume $a < $c < $b and $fa.$lt0() and $fb.$gt0()
    ($a: ident $b: ident $c: ident $d: ident,
     $fa: ident $fb: ident $fc: ident $fd: ident, $self: ident, $x: ident,
     $assign: ident, $ok_val: ident, $lt0: ident, $gt0: ident) => {
        if $fc.$lt0() {
            if $self.t.stop(&$c, &$b, &$fc, &$fb) {
                $x.assign_mid(&$c, &$b);
                return Ok($ok_val!($c, $b))
            }
            $assign!($d, $a);  $assign!($fd, $fa);
            $assign!($a, $c);  $assign!($fa, $fc); // `$b` and `$fb` unchanged
        } else if $fc.$gt0() {
            if $self.t.stop(&$a, &$c, &$fa, &$fc) {
                $x.assign_mid(&$a, &$c);
                return Ok($ok_val!($a, $c))
            }
            $assign!($d, $b);  $assign!($fd, $fb);
            $assign!($b, $c);  $assign!($fb, $fc); // `$a` and `$fa` unchanged
        } else if $fc.is_finite() {
            $assign!(*$x, $c);
            return Ok($ok_val!($c, $c))
        } else {
            return Err(Error::NotFinite{ x: $c.clone(), fx: $fc.clone() })
        }
    }
}

macro_rules! assign_copy { ($y: expr, $x: ident) => { $y = $x } }
macro_rules! ok_copy { ($a: expr, $b: expr) => { ($a, $b) } }

/// `bracket_neg_pos!(a b c d, fa fb fc fd, self, x)`: update `a`,
/// `b`, and `d` (and the corresponding `fa`, `fb` and `fd`) according
/// to the sign of `fc`.
/// Assume f(a).`$lt0()` and f(b).$gt0()`.  The same invariant is true
/// on exit.
macro_rules! bracket_copy {
    ($a: ident $b: ident $c: ident $d: ident,
     $fa: ident $fb: ident $fc: ident $fd: ident,
     $self: ident, $x: ident, $lt0: ident, $gt0: ident) => {
        bracket_sign!($a $b $c $d, $fa $fb $fc $fd, $self, $x,
                      assign_copy, ok_copy, $lt0, $gt0)
    }
}

macro_rules! abs_lt_neg_pos { ($x: expr, $y: expr) => { - $x < $y } }
macro_rules! abs_lt_pos_neg { ($x: expr, $y: expr) => { $x < - $y } }

impl<T, F, Term> Toms748<T, F, Term>
where T: OrdField,
      F: FnMut(T) -> T,
      Term: Terminate<T> {
    /// Use the Algorithm 748 to approximate a root of the function
    /// `f` on the interval \[`a`, `b`\] (see [`toms748`]).  Store
    /// this approximation in `root` and return an interval \[a,b\]
    /// such that `f`(a) * `f`(b) ≤ 0 (containing `root`) and
    /// satisfying the termination criterion [`Terminate`].
    ///
    /// Note that the above description assumes that `f` is
    /// continuous.  If it is not, the description of the returned
    /// interval still holds but it is not guaranteed that `f`
    /// possesses a root in it.
    ///
    /// The error [`Error::NotFinite`] is returned if the function `f`
    /// produces a NaN value.  When [`maxiter_err`][Toms748::maxiter_err]
    /// is turned on, the error [`Error::MaxIter`] is returned if
    /// the maximum number of iterations is reached.
    #[must_use]
    pub fn root_mut(&mut self, root: &mut T) -> Result<(T,T), Error<T>> {
        let mut a;
        let mut b;
        if self.a <= self.b {
            a = self.a;
            b = self.b;
        } else {
            a = self.b;
            b = self.a;
        };
        let mut fa = (self.f)(a);
        let mut fb = (self.f)(b);
        // a ≤ b, `a` and `b` finite by construction
        if self.t.stop(&a, &b, &fa, &fb) {
            root.assign_mid(&a, &b);
            return Ok((a,b))
        }
        let mut d;
        let mut fd;
        let mut e;
        let mut fe;
        // The state is (a, b, d, e) together with the values of `f`
        // at these points.
        macro_rules! body {
            ($lt0: ident, $gt0: ident, $abs_lt: ident) => {
                // 4.2.1 = 4.1.1: (a, b) = (a₁, b₁)
                let mut c1 = a - (fa / (fb - fa)) * (b - a);
                if !Self::is_inside_interval(c1, a, b) {
                    c1.assign_mid(&a, &b);
                }
                // 4.2.2 = 4.1.2: (a, b, d) = (a₂, b₂, d₂)
                let fc1 = (self.f)(c1);
                bracket_copy!(a b c1 d, fa fb fc1 fd, self, root, $lt0, $gt0);
                // 4.2.3: n = 2
                let c2 = Self::newton_quadratic::<1>(a, b, d, fa, fb, fd);
                body!(step, c2, $lt0, $gt0, $abs_lt);
                // n = 3..
                for _ in 2 .. self.maxiter {
                    // 4.2.3: (a, b, d, e) = (aₙ, bₙ, dₙ, eₙ)
                    let mut c = Self::ipzero(a, b, d, e, fa, fb, fd, fe);
                    if !Self::is_inside_interval(c, a, b) {
                        c = Self::newton_quadratic::<1>(a, b, d, fa, fb, fd);
                    };
                    body!(step, c, $lt0, $gt0, $abs_lt);
                }
            };
            // Assume (a, b, d) = (aₙ, bₙ, dₙ) and (fa, fb, fd) =
            // (f(aₙ), f(bₙ), f(dₙ)), take cₙ, and update the state.
            (step, $c: ident, $lt0: ident, $gt0: ident, $abs_lt: ident) => {
                let dist_an_bn = b - a;
                // 4.2.4
                let fc = (self.f)($c);
                e = d; // ẽₙ  (eₙ no longer used)
                fe = fd; // f(ẽₙ)
                // (a, b, d) = (ãₙ, b̃ₙ, d̃ₙ)
                bracket_copy!(a b $c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.5
                let mut c = Self::ipzero(a, b, d, e, fa, fb, fd, fe);
                if !Self::is_inside_interval(c, a, b) {
                    c = Self::newton_quadratic::<2>(a, b, d, fa, fb, fd);
                };
                // 4.2.6: (a, b, d) = (a̅ₙ, b̅ₙ, d̅ₙ)
                let fc = (self.f)(c);
                bracket_copy!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.7 = 4.1.5: u = uₙ
                debug_assert!(fa.$lt0() && fb.$gt0());
                let len = b - a;
                let mut c;
                let mut dist;
                if $abs_lt!(fa, fb) {
                    // 4.2.8 = 4.1.6: c = c̅ₙ with uₙ = a
                    dist = - fa.twice() / (fb - fa) * len;
                    c = a + dist;
                    if c <= a { // `dist` may be very "absorbed" by `a`
                        dist = len.div64();
                        c = a + dist
                    }
                } else {
                    // 4.2.8 = 4.1.6: c = c̅ₙ with uₙ = b
                    dist = (fb.twice() / (fb - fa)) * len;
                    c = b - dist;
                    if c >= b {
                        dist = len.div64();
                        c = b - dist
                    }
                };
                // 4.2.9 = 4.1.7: c = ĉₙ
                if dist.twice() > len {
                    c.assign_mid(&a, &b);
                }
                // 4.2.10 = 4.1.8: (a, b, d) = (âₙ, b̂ₙ, d̂ₙ)
                let fc = (self.f)(c);
                e = d; // save d̅ₙ and anticipate eₙ₊₁ = d̅ₙ
                fe = fd;
                bracket_copy!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.11 = 4.1.9
                // Nothing to do for the first case.
                if (b - a).twice() >= dist_an_bn { // μ = 1/2
                    e = d;  fe = fd; // eₙ₊₁ = d̂ₙ
                    c.assign_mid(&a, &b); // reuse `c`
                    let fmid = (self.f)(c);
                    bracket_copy!(a b c d, fa fb fmid fd, self, root,
                                  $lt0, $gt0);
                }
            }
        }
        match check_sign("root1d::toms748", a, b, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0, abs_lt_neg_pos); }
            SignChange::PosNeg => { body!(gt0, lt0, abs_lt_pos_neg); },
            SignChange::Root1 => { *root = a;  return Ok((a,a)) }
            SignChange::Root2 => { *root = b;  return Ok((b,b)) }
        }
        if self.maxiter_err {
            return Err(Error::MaxIter)
        }
        root.assign_mid(&a, &b);
        Ok((a,b))
    }

    /// Evaluate with `K`+1 Newton iterations the root of the quadratic
    /// interpolation polynomial on (x, f(x)) with x ∈ {a, b, d}.
    /// Will only be used with `a <= b`.
    #[inline]
    #[must_use]
    fn newton_quadratic<const K: u8>(a: T, b: T, d: T,
                                     fa: T, fb: T, fd: T) -> T {
        let fab = (fa - fb) / (a - b);
        let fbd = (fb - fd) / (b - d);
        let fabd = (fab - fbd) / (a - d);
        let den = fab - fabd * (a + b);
        let (mut r, mut p) = if (fabd * fa).gt0() { (a, fa) } else { (b, fb) };
        for _ in 0..K {
            r = r - p / (den + fabd * r.twice());
            p = fa + fab * (r - a) + fabd * (r - a) * (r - b);
        }
        r = r - p / (den + fabd * r.twice());
        if Self::is_inside_interval(r, a, b) {
            r
        } else { // Maybe fabd = 0, or d ∈ {a,b},...
            let r = a - fa / fab;
            if r <= a { a + (b - a).div64() }
            else if r >= b { b - (b - a).div64() }
            else { r }
        }
    }

    /// Return `true` if `x` ∈ \]`a`, `b`\[.  If `x` is NaN or ±∞
    /// (coming, say, from a division by 0), this function returns
    /// `false`.  Assume that `a` ≤ `b`.
    #[inline]
    fn is_inside_interval(x: T, a: T, b: T) -> bool {
        a < x && x < b
    }

    /// Compute IP(0), the value at 0 of the inverse cubic interporation.
    #[inline]
    #[must_use]
    fn ipzero(a: T, b: T, c: T, d: T, fa: T, fb: T, fc: T, fd: T) -> T {
        // See “J. Stoer and R. Bulirsch, Introduction to numerical
        // analysis, 3rd ed. New York: Springer, 2002”, p. 43.
        let a_b = (a - b) / (fb - fa);
        let d31 = a_b * fb;
        let q31 = a_b * fa;
        let b_c = (b - c) / (fc - fb);
        let d21 = b_c * fc;
        let q21 = b_c * fb;
        let d31_q21 = (d31 - q21) / (fc - fa);
        let d32 = d31_q21 * fc;
        let q32 = d31_q21 * fa;
        let q11 = (c - d) * fc / (fd - fc);
        let q22 = (d21 - q11) * fb / (fd - fb);
        let q33 = (d32 - q22) * fa / (fd - fa);
        a + (q31 + q32 + q33)
    }

    /// Return `Ok(r)` where `r` is a root of the function or `Err` if
    /// the algorithm did not converge.  See
    /// [`root_mut`][Toms748::root_mut] for more information.
    #[must_use]
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut x = self.a;
        self.root_mut(&mut x).and(Ok(x))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][Toms748::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(T,T), Error<T>> {
        let mut x = self.a;
        self.root_mut(&mut x)
    }

}

////////////////////////////////////////////////////////////////////////
//
// Toms 748 for non-copy types

/// Requirements on the type `T` to be able to use [`toms748_mut`]
/// algorithm.
pub trait OrdFieldMut: Bisectable
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self> {
        /// Return `true` if `self` is the number 0.
        fn is_zero(&self) -> bool;

        /// Multiply in place `self` by 2.
        fn twice(&mut self);

        /// Perform the negation.
        fn neg_assign(&mut self);

        /// Return `true` if `self` and `other` have the same sign and
        /// are both non-zero.
        #[inline]
        fn has_same_sign(&self, other: &Self) -> bool {
            (self.gt0() && other.gt0()) || (self.lt0() && other.lt0())
        }
    }

new_root_finding_method!(
    /// Same as [`toms748`] for non-[`Copy`] types.
    toms748_mut,
    /// [`toms748_mut`] algorithm (for non-[`Copy`] types).
    Toms748Mut<'a,...>,
    workspace, Option<&'a mut (T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T)>);

impl<'a, T, F, Term> Toms748Mut<'a, T, F, Term>
where T: OrdFieldMut,
      Term: Terminate<T> {
    /// Provide variables that will be used as workspace when running
    /// the [`toms748_mut`] function.
    #[must_use]
    pub fn work(mut self, w: &'a mut (T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T)) -> Self {
        self.workspace = Some(w);
        self
    }
}

macro_rules! assign_mut { ($y: expr, $x: ident) => { $y.assign($x) } }
macro_rules! ok_mut { ($a: expr, $b: expr) => { () } }

/// `bracket_neg_pos!(a b c d, fa fb fc fd, self, x)`: update `a`,
/// `b`, and `d` (and the corresponding `fa`, `fb` and `fd`) according
/// to the sign of `fc`.
/// Assume f(a).`$lt0()` and f(b).$gt0()`.  The same invariant is true
/// on exit.
macro_rules! bracket_mut {
    ($a: ident $b: ident $c: ident $d: ident,
     $fa: ident $fb: ident $fc: ident $fd: ident,
     $self: ident, $x: ident, $lt0: ident, $gt0: ident) => {
        bracket_sign!($a $b $c $d, $fa $fb $fc $fd, $self, $x,
                      assign_mut, ok_mut, $lt0, $gt0)
    }
}

macro_rules! abs_lt_neg_pos_mut {
    ($x: ident, $y: ident, tmp = $t: ident) => {{
        $t.assign($x);  $t.neg_assign();   $t < $y
    }}
}

macro_rules! abs_lt_pos_neg_mut {
    ($x: ident, $y: ident, tmp = $t: ident) => {{
        $t.assign($y);  $t.neg_assign();   $x < $t
    }}
}

impl<'a, T, F, Term> Toms748Mut<'a, T, F, Term>
where T: OrdFieldMut + 'a,
      F: FnMut(&mut T, &T),
      Term: Terminate<T> {

    /// Return a root of the function `f` (see [`toms748_mut`]) or
    /// `Err(e)` to indicate that the function `f` returned a NaN
    /// value.
    #[must_use]
    pub fn root(&mut self) -> Result<T, Error<T>> {
        let mut root = self.a.clone();
        self.root_mut(&mut root).and(Ok(root))
    }

    /// Set `root` to a root of the function `f` (see [`toms748_mut`]).
    /// Return `Some(err)` to indicate that the algorithm failed
    /// (e.g., when the function `f` returned a NaN value).
    ///
    /// If the [`work`][`Toms748Mut::work`] method was not used,
    /// internal variables are constructed by cloning `root`, thereby
    /// inheriting its precision for example.
    #[must_use]
    pub fn root_mut(&mut self, root: &mut T) -> Result<(), Error<T>> {
        let mut tmp;
        let (a, b, c, d, e, fa, fb, fc, fd, fe,
             t1, t2, t3, t4, t5, dist_an_bn) = match &mut self.workspace {
            None => {
                tmp = (root.clone(), root.clone(), root.clone(), root.clone(),
                       root.clone(), root.clone(), root.clone(), root.clone(),
                       root.clone(), root.clone(), root.clone(), root.clone(),
                       root.clone(), root.clone(), root.clone(), root.clone());
                (&mut tmp.0, &mut tmp.1, &mut tmp.2, &mut tmp.3,
                 &mut tmp.4, &mut tmp.5, &mut tmp.6, &mut tmp.7,
                 &mut tmp.8, &mut tmp.9, &mut tmp.10, &mut tmp.11,
                 &mut tmp.12, &mut tmp.13, &mut tmp.14, &mut tmp.15)
            }
            Some(v) =>
                (&mut v.0, &mut v.1, &mut v.2, &mut v.3,
                 &mut v.4, &mut v.5, &mut v.6, &mut v.7,
                 &mut v.8, &mut v.9, &mut v.10, &mut v.11,
                 &mut v.12, &mut v.13, &mut v.14, &mut v.15)
        };
        if self.a <= self.b {
            a.assign(self.a);
            b.assign(self.b);
        } else {
            a.assign(self.b);
            b.assign(self.a);
        }
        // a ≤ b, `a` and `b` finite by construction
        macro_rules! body {
            ($lt0: ident, $gt0: ident, $abs_lt: ident) => {
                // `fa` and `fb` set by `check_sign_mut`.
                if self.t.stop(&a, &b, &fa, &fb) {
                    root.assign_mid(a, b);
                    return Ok(())
                }
                // 4.2.1 = 4.1.1: (a, b) = (a₁, b₁)
                c.assign(a);
                t1.assign(b);  *t1 -= a;  *t1 *= fa;
                t2.assign(fb);  *t2 -= fa;  *t1 /= t2;
                *c -= t1; // c = a - (fa / (fb - fa)) * (b - a);
                if !Self::is_inside_interval(&c, &a, &b) {
                    c.assign_mid(&a, &b);
                }
                // 4.2.2 = 4.1.2: (a, b, d) = (a₂, b₂, d₂)
                (self.f)(fc, c);
                bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.3: n = 2
                Self::newton_quadratic::<1>(c, [t1, t2, t3, t4, t5],
                                            a, b, d, fa, fb, fd);
                body!(step, $lt0, $gt0, $abs_lt);
                // n = 3..
                for _ in 2 .. self.maxiter {
                    // 4.2.3: (a, b, d, e) = (aₙ, bₙ, dₙ, eₙ)
                    Self::ipzero(c, [t1, t2, t3, t4],
                                 a, b, d, e, fa, fb, fd, fe);
                    if !Self::is_inside_interval(&c, &a, &b) {
                        Self::newton_quadratic::<1>(c, [t1, t2, t3, t4, t5],
                                                    a, b, d, fa, fb, fd);
                    };
                    body!(step, $lt0, $gt0, $abs_lt);
                }
            };
            // Assume (a, b, c, d) = (aₙ, bₙ, cₙ, dₙ) and (fa, fb, fd) =
            // (f(aₙ), f(bₙ), f(dₙ)), take cₙ, and update the state.
            (step, $lt0: ident, $gt0: ident, $abs_lt: ident) => {
                dist_an_bn.assign(b); *dist_an_bn -= a; // b - a
                // 4.2.4
                (self.f)(fc, c);
                e.assign(d); // ẽₙ  (eₙ no longer used)
                fe.assign(fd); // f(ẽₙ)
                // (a, b, d) = (ãₙ, b̃ₙ, d̃ₙ)
                bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.5
                Self::ipzero(c, [t1, t2, t3, t4], a, b, d, e, fa, fb, fd, fe);
                if !Self::is_inside_interval(c, a, b) {
                    Self::newton_quadratic::<2>(c, [t1, t2, t3, t4, t5],
                                                a, b, d, fa, fb, fd);
                };
                // 4.2.6: (a, b, d) = (a̅ₙ, b̅ₙ, d̅ₙ)
                (self.f)(fc, c);
                bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.7 = 4.1.5: c = uₙ, t1 = f(uₙ)
                debug_assert!(fa.$lt0() && fb.$gt0());
                if $abs_lt!(fa, fb, tmp = t1) {
                    c.assign(a);  t1.assign(fa);
                } else {
                    c.assign(b);  t1.assign(fb);
                };
                // 4.2.8 = 4.1.6
                t1.twice();
                t2.assign(b);  *t2 -= a;  *t1 *= t2;
                t2.assign(fb);  *t2 -= fa;  *t1 /= t2; // t1 = uₙ - c̅ₙ
                *c -= t1; // c = c̅ₙ = uₙ - 2 f(uₙ) * (b - a) / (fb - fa)
                // 4.2.9 = 4.1.7: c = ĉₙ
                t1.twice(); // t1 = 2(uₙ - c̅ₙ)
                if t1.lt0() { t1.neg_assign() }; // t1 = 2 |c̅ₙ - uₙ|
                t2.assign(b);  *t2 -= a; // t2 = b - a > 0
                if t1 > t2 {
                    c.assign_mid(&a, &b);
                }
                // 4.2.10 = 4.1.8: (a, b, d) = (âₙ, b̂ₙ, d̂ₙ)
                (self.f)(fc, c);
                e.assign(d); // save d̅ₙ and anticipate eₙ₊₁ = d̅ₙ
                fe.assign(fd);
                bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.11 = 4.1.9
                // Nothing to do for the first case.
                t1.assign(b);  *t1 -= a;  t1.twice(); // t1 = 2(b - a)
                if t1 >= dist_an_bn { // μ = 1/2
                    e.assign(d);  fe.assign(fd); // eₙ₊₁ = d̂ₙ
                    c.assign_mid(&a, &b); // reuse `c`
                    (self.f)(fc, c);
                    bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                }
            }
        }
        match check_sign_mut("root1d::toms748_mut", a, b,
                             &mut self.f, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0, abs_lt_neg_pos_mut); }
            SignChange::PosNeg => { body!(gt0, lt0, abs_lt_pos_neg_mut); },
            SignChange::Root1 => { root.assign(a); return Ok(()) },
            SignChange::Root2 => { root.assign(b); return Ok(())},
        }
        if self.maxiter_err {
            return Err(Error::MaxIter)
        }
        root.assign_mid(&a, &b);
        Ok(())
    }

    /// Return `true` if `x` ∈ \]`a`, `b`\[.  If `x` is NaN or ±∞
    /// (coming, say, from a division by 0), this function returns
    /// `false`.  Assume that `a` ≤ `b`.
    #[inline]
    fn is_inside_interval(x: &T, a: &T, b: &T) -> bool {
        a < x && x < b
    }

    /// Evaluate with `K`+1 Newton iterations the root of the quadratic
    /// interpolation polynomial on (x, f(x)) with x ∈ {a, b, d}.
    #[inline]
    fn newton_quadratic<const K: u8>(
        r: &mut T, [fab, fabd, t1, den, p]: [&mut T; 5],
        a: &T, b: &T, d: &T,  fa: &T, fb: &T, fd: &T) {
        fab.assign(fa);  *fab -= fb; // fa - fb
        t1.assign(a);  *t1 -= b; // a - b
        *fab /= t1; // fab = (fa - fb) / (a - b)
        fabd.assign(fb);  *fabd -= fd;
        t1.assign(b);  *t1 -= d;
        *fabd /= t1; // f[b,d] = (fb - fd) / (b - d)
        *fabd -= fab;
        t1.assign(d);  *t1 -= a;
        *fabd /= t1; // fabd = (fab - fbd) / (a - d)
        den.assign(fab);
        t1.assign(a);  *t1 += b;  *t1 *= fabd;
        *den -= t1; // den = fab - fabd * (a + b)
        if fabd.has_same_sign(&fa) {
            r.assign(a); p.assign(fa)
        } else {
            r.assign(b); p.assign(fb)
        }
        macro_rules! update {
            (r) => { // also change `p`
                t1.assign(r);  t1.twice();  *t1 *= fabd;  *t1 += den;
                *p /= t1;
                *r -= p; // r := r - p / (den + fabd * 2r)
            };
        }
        for _ in 0..K {
            update!(r);
            // Update `p` (do not change `r`)
            p.assign(fabd);
            t1.assign(r);  *t1 -= b;  *p *= t1;
            *p += fab; // p = fab + fabd * (r - b)
            t1.assign(r);  *t1 -= a;  *p *= t1;
            *p += fa; // fa + fab * (r - a) + fabd * (r - a) * (r - b)
        }
        update!(r);
        if !Self::is_inside_interval(r, a, b) {
            r.assign(a);
            t1.assign(fa);  *t1 /= fab;
            *r -= t1; // a - fa / fab
        }
    }

    /// Compute IP(0), the value at 0 of the inverse cubic interporation.
    #[inline]
    fn ipzero(r: &mut T, [t1, t2, t3, t4]: [&mut T; 4],
              a: &T, b: &T, c: &T, d: &T,
              fa: &T, fb: &T, fc: &T, fd: &T) {
        // See the implementation of `ipzero` for Copy types.
        r.assign(a);  *r -= b;
        t1.assign(fb);  *t1 -= fa;  *r /= t1; // r = (a - b) / (fb - fa)
        t1.assign(fb);  *t1 *= r; // t1 = d31 = (a - b) / (fb - fa) * fb
        *r *= fa; // r = q31
        t2.assign(b);  *t2 -= c;
        t3.assign(fc);  *t3 -= fb;  *t2 /= t3; // t2 = b_c = (b - c) / (fc - fb)
        t3.assign(fc);  *t3 *= t2; // t3 = d21
        *t2 *= fb; // t2 = q21
        *t1 -= t2; // t1 = d31 - q21  (t2 is free)
        t2.assign(fc);  *t2 -= fa;  *t1 /= t2; // t1 = (d31 - q21) / (fc - fa)
        t2.assign(fc);  *t2 *= t1; // t2 = d32
        *t1 *= fa; // t1 = q32
        *r += t1; // r = q31 + q32  (t1 is free)
        t1.assign(c);  *t1 -= d;  *t1 *= fc;
        t4.assign(fd);  *t4 -= fc;  *t1 /= t4; // t1 = q11 = (c - d) * fc / (fd - fc)
        *t3 -= t1;  *t3 *= fb; // t3 = (d21 - q11) * fb  (t1 free)
        t1.assign(fd);  *t1 -= fb;  *t3 /= t1; // t3 = q22
        *t2 -= t3;  *t2 *= fa; // t2 = (d32 - q22) * fa
        t1.assign(fd);  *t1 -= fa;  *t2 /= t1; // t2 = q33
        *r += t2;
        *r += a;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Compatibility with Rug

#[cfg(feature = "rug")]
mod rug {
    use std::{cell::RefCell,
              cmp::Ordering};
    use rug::{Float, Rational, float::Round, ops::AssignRound};
    use crate::{Bisectable, Terminate, Tol, OrdFieldMut};

    /// Termination criterion based on tolerances for Rug.
    pub struct TolRug<T> {
        rtol: T,
        atol: T,
        tmp1: RefCell<T>,
        tmp2: RefCell<T>,
    }

    macro_rules! impl_rug {
        ($t: ty, $new_t: ident, $rtol: expr, $atol: expr,
         $is_finite: ident, $assign_mid: item, $is_zero: ident,
         $less: expr) => {
            impl Default for TolRug<$t> {
                fn default() -> Self {
                    TolRug { rtol: $rtol,
                             atol: $atol,
                             tmp1: RefCell::new($new_t!()),
                             tmp2: RefCell::new($new_t!()), }
                }
            }
            impl Terminate<$t> for TolRug<$t> {
                #[inline]
                fn stop(&mut self, a: &$t, b: &$t, _: &$t, _: &$t) -> bool {
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
                    tmp1.assign(b);
                    *tmp1 -= a; // b - a
                    tmp1 <= tmp2
                }
            }

            impl Bisectable for $t {
                type DefaultTerminate = TolRug<Self>;
                #[inline]
                fn lt0(&self) -> bool { *self < 0u8 }
                #[inline]
                fn gt0(&self) -> bool { *self > 0u8 }
                #[inline]
                fn is_finite(&self) -> bool { $is_finite!(self) }
                #[inline]
                fn assign(&mut self, rhs: &Self) {
                    <Self as rug::Assign<&Self>>::assign(self, rhs); }
                #[inline]
                $assign_mid
            }

            impl OrdFieldMut for $t {
                #[inline]
                fn is_zero(&self) -> bool { $is_zero!(self) }

                #[inline]
                fn twice(&mut self) { *self *= 2 }

                #[inline]
                fn neg_assign(&mut self) {
                    <Self as rug::ops::NegAssign>::neg_assign(self)
                }
            }
        }
    }

    macro_rules! float_new { () => { Float::new(53) } }
    macro_rules! float_is_finite { ($s: expr) => { Float::is_finite($s) } }
    macro_rules! float_is_zero { ($s: expr) => { Float::is_zero($s) } }
    impl_rug!(Float, float_new,
              Float::with_val(53, 1e-16),
              Float::with_val(53, 1e-12),
              float_is_finite,
              fn assign_mid(&mut self, a: &Self, b: &Self) {
                  self.assign(a);
                  *self += b;
                  *self /= 2i8;
              },
              float_is_zero,
              Some(Ordering::Less));

    impl<U> From<Tol<U>> for TolRug<Float>
    where Float: AssignRound<U, Round = Round, Ordering = Ordering> {
        fn from(t: Tol<U>) -> Self {
            let mut rtol = float_new!();
            rtol.assign_round(t.rtol, Round::Nearest);
            let mut atol = float_new!();
            atol.assign_round(t.atol, Round::Nearest);
            TolRug { rtol, atol,
                     tmp1: RefCell::new(float_new!()),
                     tmp2: RefCell::new(float_new!()), }
        }
    }

    impl<U> super::SetTolerances<U> for TolRug<Float>
    where Float: AssignRound<U, Round = Round, Ordering = Ordering> {
        fn set_rtol(&mut self, rtol: U) {
            self.rtol.assign_round(rtol, Round::Nearest);
            if self.rtol <= 0 {
                panic!("root1d<rug::Float>: rtol = {:e} ≤ 0", self.rtol)
            }
        }
        fn set_atol(&mut self, atol: U) {
            self.atol.assign_round(atol, Round::Nearest);
            if self.atol < 0 {
                panic!("root1d<rug::Float>: atol = {:e} < 0", self.atol)
            }
        }
    }

    macro_rules! rational_new { () => { Rational::new() } }
    macro_rules! rational_is_finite { ($s: ident) => { true } }
    macro_rules! rational_is_zero { ($s: ident) => {
        $s.cmp0() == Ordering::Equal
    } }
    impl_rug!(Rational, rational_new,
              (1, 1000_0000_0000_0000u64).into(),
              (1, 1000_0000_0000_0000u64).into(),
              rational_is_finite,
              fn assign_mid(&mut self, a: &Self, b: &Self) {
                  self.assign(a);
                  *self += b;
                  *self /= 2u8;
              },
              rational_is_zero,
              Ordering::Less);

    impl<U> From<Tol<U>> for TolRug<Rational>
    where Rational: rug::Assign<U> {
        fn from(t: Tol<U>) -> Self {
            let mut rtol = rational_new!();
            <Rational as rug::Assign<U>>::assign(&mut rtol, t.rtol);
            let mut atol = rational_new!();
            <Rational as rug::Assign<U>>::assign(&mut atol, t.atol);
            TolRug { rtol, atol,
                     tmp1: RefCell::new(rational_new!()),
                     tmp2: RefCell::new(rational_new!()), }
        }
    }

    impl<U> super::SetTolerances<U> for TolRug<Rational>
    where Rational: rug::Assign<U> {
        fn set_rtol(&mut self, rtol: U) {
            <Rational as rug::Assign<U>>::assign(&mut self.rtol, rtol);
            if self.rtol <= 0 {
                panic!("root1d<rug::Rational>: rtol = {} ≤ 0", self.rtol)
            }
        }
        fn set_atol(&mut self, atol: U) {
            <Rational as rug::Assign<U>>::assign(&mut self.atol, atol);
            if self.atol < 0 {
                panic!("root1d<rug::Rational>: atol = {} < 0", self.atol)
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
// Tests

#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a: expr, $b: expr, $err: expr) => {
        let a = $a;
        let b = $b;
        if ! ((a.clone() - b.clone()).abs() <= $err) {
            panic!("|left - right| ≤ {:e}\n  left: {}\n right: {}",
                   $err, a, b);
        }
    }
}


#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use crate as root1d;
    //use test::bench::Bencher;

    type R<T> = Result<(), root1d::Error<T>>;

    #[test]
    fn bisect_decr_bounds() -> R<f64> {
        let f = |x| x * x - 2.;
        assert_approx_eq!(root1d::bisect(f, 2., 0.).maxiter_err(true).root()?,
                          2f64.sqrt(), 1e-12);
        assert_approx_eq!(root1d::bisect(f, 0., -2.).maxiter_err(true).root()?,
                          - 2f64.sqrt(), 1e-12);
        Ok(())
    }

    #[test]
    fn test_ipzero() {
        // y³ + 3 = x
        let r = root1d::Toms748::<f64, fn(f64) -> f64, root1d::Tol<f64>>
            ::ipzero(-5., 2., 4., 11.,
                     -2., -1., 1., 2.);
        assert_eq!(r, 3.);
    }

    #[test]
    fn test_newton_quadratic() {
        let r = root1d::Toms748::<f64, fn(f64) -> f64, root1d::Tol<f64>>
            ::newton_quadratic::<0>(0.,  2., 1.,
                                    -1., 3., 0.);
        assert_eq!(r, 1.25);
        let r = root1d::Toms748::<f64, fn(f64) -> f64, root1d::Tol<f64>>
            ::newton_quadratic::<1>(0.,  2., 1.,
                                    -1., 3., 0.);
        assert_eq!(r, 41. / 40.);
    }

    #[test]
    fn toms748() -> R<f64> {
        for i in 2 .. 20 {
            let c = i as f64;
            let f = |x| x * x - c;
            assert_approx_eq!(
                root1d::toms748(f, 0., c).atol(0.).rtol(1e-15).root()?,
                c.sqrt(), 1e-15);
        }
        Ok(())
    }

    #[test]
    fn toms748_large_interval() -> R<f64> {
        let f = |x| x * x - 2.;
        let r = root1d::toms748(f, 1., 1e60).maxiter(130).root()?;
        assert_approx_eq!(r, 2f64.sqrt(), 1e-15);
        Ok(())
    }

    #[test]
    fn toms748_no_term() -> R<f64> {
        let f = |x| x * x - 2.;
        let stop = |_: &f64, _: &f64, _: &f64, _: &f64| false;
        let r = root1d::toms748(f, 1., 1e60).terminate(stop).root()?;
        assert_approx_eq!(r, 2f64.sqrt(), 1e-15);
        Ok(())
    }

    #[test]
    fn bisect_discontinuous() -> R<f64> {
        let f = |x:f64| 1f64.copysign(x.sin());
        let r = root1d::bisect(f, 3., 4.).root()?;
        assert_approx_eq!(r, PI, 1e-12);
        Ok(())
    }

    #[test]
    fn toms748_discontinuous() -> R<f64> {
        let f = |x:f64| 1f64.copysign(x.sin());
        let r = root1d::toms748(f, 3., 4.).root()?;
        assert_approx_eq!(r, PI, 1e-12);
        Ok(())
    }

    // #[bench]
    // fn bisection_f64_speed(b: &mut Bencher) {
    //     let f = |x| x * x - 2.;
    //     b.iter(|| root1d::toms748(f, 0., 100.));
    // }
}

#[cfg(all(test, feature = "rug"))]
mod tests_rug {
    use std::f64::consts::PI;
    use crate as root1d;
    use rug::{Assign, Float, Rational};

    type R<T> = Result<(), root1d::Error<T>>;

    #[test]
    fn bisect_decr_bounds() -> R<Float> {
        let f = |y: &mut Float, x: &Float| {
            y.assign(x*x);
            *y -= 2.;
        };
        let a = Float::with_val(53, -2_f64);
        let b = Float::with_val(53, 0f64);
        let c = Float::with_val(53, 2_f64);
        assert_approx_eq!(
            root1d::bisect_mut(f, &c, &b).maxiter_err(true).root()?.to_f64(),
            2f64.sqrt(), 1e-12);
        assert_approx_eq!(
            root1d::bisect_mut(f, &b, &a).maxiter_err(true).root()?.to_f64(),
            - 2f64.sqrt(), 1e-12);
        Ok(())
    }

    #[test]
    fn bisection() -> R<Float> {
        for i in 2 .. 20 {
            let a = Float::with_val(53, 0f64);
            let b = Float::with_val(53, i as f64);
            let f = |y: &mut Float, x: &Float| {
                y.assign(x*x);
                *y -= &b;
            };
            assert_approx_eq!(
                root1d::bisect_mut(f, &a, &b).atol(1e-15).root()?,
                b.sqrt(), 1e-15);
        }
        Ok(())
    }

    #[test]
    fn bisection_rational() -> R<Rational> {
        for i in 2 .. 20 {
            let a = Rational::from(0);
            let b = Rational::from(i);
            let f = |y: &mut Rational, x: &Rational| {
                y.assign(x*x);
                *y -= &b;
            };
            assert_approx_eq!(root1d::bisect_mut(f, &a, &b).root()?.to_f64(),
                              (i as f64).sqrt(), 3e-15);
        }
        Ok(())
    }

    #[test]
    fn toms748() -> R<Float> {
        for i in 2 .. 20 {
            let a = Float::with_val(53, 0f64);
            let b = Float::with_val(53, i as f64);
            let f = |y: &mut Float, x: &Float| {
                y.assign(x * x);
                *y -= &b;
            };
            assert_approx_eq!(
                root1d::toms748_mut(f, &a, &b).atol(1e-15).root()?,
                b.sqrt(), 1e-15);
        }
        Ok(())
    }

    #[test]
    fn toms748_rational() -> R<Rational> {
        for i in 2 .. 20 {
            let a = Rational::from(0);
            let b = Rational::from(i);
            let f = |y: &mut Rational, x: &Rational| {
                y.assign(x*x);
                *y -= &b;
            };
            assert_approx_eq!(root1d::toms748_mut(f, &a, &b).root()?.to_f64(),
                              (i as f64).sqrt(), 1e-15);
        }
        Ok(())
    }

    #[test]
    fn bisect_discontinuous() -> R<Float> {
        let a = Float::with_val(53, 3f64);
        let b = Float::with_val(53, 4f64);
        let f = |y: &mut Float, x: &Float| {
            y.assign(1.);
            y.copysign_mut(&Float::with_val(53, x.sin_ref()))
        };
        assert_approx_eq!(root1d::bisect_mut(f, &a, &b).root()?.to_f64(),
                          PI, 1e-12);
        Ok(())
    }
}
