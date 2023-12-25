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
//! All root finding procedures have the same structure.  For example
//! algorithm Toms 748 starts with a function [`toms748`] to specify
//! the function `f` and the interval \[`a`, `b`\] on which `f`
//! changes sign.  It returns a structure [`Toms748`] which provides
//! methods to specify various parameters (such as
//! [`rtol`][Toms748::rtol]) and functions to compute a root (such as
//! [`root`][Toms748::root] and [`root_mut`][Toms748::root_mut]).
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use root1d::toms748;
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
//! [`toms748`] (resp. [`toms748_mut`]), you must also implement the
//! trait [`OrdField`] (resp. [`OrdFieldMut`]).

#![cfg_attr(feature = "nightly", feature(never_type))]

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    mem::swap,
    ops::{
        Neg, Add, Sub, Mul, Div,
        AddAssign, SubAssign, MulAssign, DivAssign
    },
    marker::PhantomData,
    result::Result,
};

#[cfg(feature = "nightly")]
/// Type indicating that no error is raised by the function.
pub type NoError = !;
#[cfg(not(feature = "nightly"))]
#[derive(Debug)]
/// Type indicating that no error is raised by the function.
pub enum NoError {}

/// Errors that may be returned by the root finding methods.
#[derive(Debug)]
pub enum Error<T, E = NoError> {
    /// Error indicating that the function evaluated at `x` returned
    /// the non-finite value `fx`.
    NotFinite { x: T, fx: T },
    /// Error returned by the function evaluated at `x`.
    Fun { x: T, err: E },
    /// Report that the maximum number of iterations has been reached,
    /// when option `maxiter_err` is turned on.  The argument is the
    /// current estimate of the root at that moment.
    MaxIter,
    /// Report that the function did not change sign on the original
    /// interval \[a, b\].
    NoSignChange { a: T, fa: T, b: T, fb: T },
}
impl<T: Display, E: Debug> Display for Error<T, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFinite { x, fx } => {
                write!(f, "evaluating the function at {} yields {} which is \
                           not finite", x, fx)
            }
            Error::Fun { x, err } => {
                write!(f, "evaluating the function at {} yields the error \
                          {:?}", x, err)
            }
            Error::MaxIter => {
                write!(f, "maximum number of iterations reached")
            }
            Error::NoSignChange {a, fa, b, fb } =>
                write!(f, "No change of sign on [a, b] = [{a}, {b}], \
                           f(a) = {fa}, f(b) = {fb}"),
        }
    }
}

impl<T: Debug + Display, E: Debug> std::error::Error for Error<T, E> {}

////////////////////////////////////////////////////////////////////////
//
// Error returning functions

/// A float type or a Result type for floats.  This is intended for
/// copy types.
pub trait FloatOrError<T> {
    type Error;
    fn to_result(self) -> Result<T, Self::Error>;
}

macro_rules! impl_float_or_error { ($t: ty) => {
    impl FloatOrError<$t> for $t {
        type Error = NoError;

        #[inline]
        fn to_result(self) -> Result<$t, NoError> { Ok(self) }
    }
    impl<E> FloatOrError<$t> for Result<$t, E> {
        type Error = E;

        #[inline]
        fn to_result(self) -> Result<$t, E> { self }
    }
}}

impl_float_or_error!(f64);
impl_float_or_error!(f32);

#[inline]
fn eval_float_result<T, F, R>(mut f: F, x: T) -> Result<T, Error<T, R::Error>>
where
    T: Copy,
    F: FnMut(T) -> R,
    R: FloatOrError<T>,
{
    f(x).to_result()
        .map_err(|err| Error::Fun{ x, err })
}

/// A unit type or a `Result<(), _>`.
pub trait UnitOrError {
    type Error;
    fn to_result(self) -> Result<(), Self::Error>;
}

impl UnitOrError for () {
    type Error = NoError;

    #[inline]
    fn to_result(self) -> Result<(), Self::Error> { Ok(()) }
}

impl<E> UnitOrError for Result<(), E> {
    type Error = E;

    #[inline]
    fn to_result(self) -> Result<(), E> { self }
}

#[inline]
fn eval_unit_result<T, F, R>(
    mut f: F, y: &mut T, x: &T
) -> Result<(), Error<T, R::Error>>
where
    T: Clone,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError,
{
    f(y, x).to_result()
        .map_err(|err| Error::Fun {
            x: x.clone(), err })
}

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
    /// Set the relative tolerance.  Set the default value if `rtol` is ≤ 0.
    fn set_rtol(&mut self, rtol: U);
    /// Set the absolute tolerance.  Set the default value if `atol` is < 0.
    fn set_atol(&mut self, atol: U);
}

/// Enable using a closure `stop` as a termination criterion.  The
/// call `stop(a, b, fa, fb)` must say whether the interval
/// \[`a`, `b`\] is satisfactory (in which case it must return `true`)
/// or not.  `fa` and `fb` are the values of the fonction to which the
/// root-finding algorithm is applied at the bounds `a` and `b`
/// respectively.
impl<T,F> Terminate<T> for F
where
    F: FnMut(&T, &T, &T, &T) -> bool,
    T: Bisectable,
{
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
    pub atol: U,
}

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
                self.rtol = if rtol <= 0. { $rtol } else { rtol }
            }
            fn set_atol(&mut self, atol: $t) {
                self.atol = if atol < 0. { $atol } else { atol }
            }
        }
    }
}

impl_traits_tol_fXX!(f64, 4. * f64::EPSILON, 2e-12);
impl_traits_tol_fXX!(f32, 4. * f32::EPSILON, 2e-6);

// Update if https://rust-lang.github.io/rfcs/3453-f16-and-f128.html
// is merged.

////////////////////////////////////////////////////////////////////////
//
// Bisectable types

/// Trait indicating that the type is suitable for the bisection algorithm.
pub trait Bisectable: PartialOrd + Clone + Debug {
    /// Type for the default termination criteria.
    type DefaultTerminate: Default + Terminate<Self>;

    /// Return `true` if `self` is `< 0` (thus not a NaN).
    /// This function must be compatible with [`PartialOrd`].
    fn lt0(&self) -> bool;

    /// Return `true` if `self` is `> 0` (thus not a NaN).
    /// This function must be compatible with [`PartialOrd`].
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
    ($fun: ident,
     // The structure to hold the options (and other fields).
     $(#[$doc: meta])* $struct: ident <$($l: lifetime,)? ...>,
     $($field: ident, $t: ty),*) => {
         $(#[$doc])*
         pub struct $struct<$($l,)? T, F, Term, E>
         where Term: Terminate<T> {
             f: F,
             a: $(&$l)? T,  // `a` and `b` are the bounds of the interval.
             b: $(&$l)? T,
             error: PhantomData<E>, // Error that `f` might return
             t: Term,  // Termination criterion
             maxiter: usize,
             maxiter_err: bool,
             $($field: $t,)*
         }

         impl <$($l,)? T, F, E> $struct<$($l,)? T, F, T::DefaultTerminate, E>
         where T: Bisectable {
             /// Private constructor called by $fun with more constraints.
             #[must_use]
             fn new(f: F, a: $(&$l)? T, b: $(&$l)? T) -> Self {
                 $struct {
                     f,  a,  b,
                     error: PhantomData,
                     t: T::DefaultTerminate::default(),
                     maxiter: 100,
                     maxiter_err: false,
                     $($field: None,)*  // All extra fields are options
                 }
             }
         }

         impl <$($l,)? T, F, Term, E> $struct<$($l,)? T, F, Term, E>
         where
             T: Bisectable,
             Term: Terminate<T>,
         {
             /// Check that `a` and `b` are finite.
             #[inline]
             fn check_interval_bounds(&self) -> Result<(), Error<T, E>> {
                 if !self.a.is_finite() {
                     return Err(Error::NotFinite {
                         x: self.a.clone(), fx: self.a.clone(),
                     })
                 }
                 if !self.b.is_finite() {
                     return Err(Error::NotFinite {
                         x: self.b.clone(), fx: self.b.clone(),
                     })
                 }
                 Ok(())
             }
         }

         impl<$($l,)? T, F, Term, E> $struct<$($l,)? T, F, Term, E>
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
             pub fn terminate<Tr>(self, t: Tr) -> $struct<$($l,)? T, F, Tr, E>
             where Tr: Terminate<T> {
                 // FIXME: type changing struct updating is experimental
                 // $s { t, .. self }
                 $struct {
                     t,
                     f: self.f,  a: self.a,  b: self.b,
                     error: self.error,
                     maxiter: self.maxiter,
                     maxiter_err: self.maxiter_err,
                     $( $field: self.$field, )*
                 }
             }

             /// Set the the relative tolerance termination criterion (that
             /// implements [`SetTolerances`]), leaving unchanged the value
             /// of the absolute tolerance.
             ///
             /// Set the default value if `rtol` is ≤ 0.
             pub fn rtol<U>(mut self, rtol: U) -> Self
             where Term: SetTolerances<U> {
                 self.t.set_rtol(rtol);
                 self
             }
             /// Set the the absolute tolerance termination criterion (that
             /// implements [`SetTolerances`]), leaving unchanged the value
             /// of the relative tolerance.
             ///
             /// Set the default value if `atol` is < 0.
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

/// Find a root of the function `f` on the interval \[`a`, `b`\]
/// with finite bounds assuming `f(a)` and `f(b)` have opposite
/// signs and `f` is continuous using the bisection algorithm.
///
/// Trying to compute the root when `f(a)` and `f(b)` do *not*
/// have opposite signs will return the error [`Error::NoSignChange`].
/// If the function is not continuous, root-finding methods will still
/// compute a small interval at the boundary of which `f` changes sign
/// and return a point in it; [`Bisect::bracket`] and
/// [`Bisect::root_mut`] return the small interval.
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
/// # Examples
///
/// ```
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
/// use root1d::bisect;
/// assert!((bisect(|x| x*x - 2., 0., 2.).atol(0.).root()?
///          - 2f64.sqrt()).abs() < 1e-15);
/// # Ok(()) }
/// ```
///
/// The function we apply the bisection to may also return an error in
/// which case the bisection stops and returns the error [`Error::Fun`].
///
/// ```
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
/// use root1d::bisect;
/// let f = |a: f64| bisect(|x| x*x - a, 0., a.max(1.)).root();
/// assert!((bisect(|a| f(a).and_then(|fa| Ok(fa - 2.)), 1., 5.).root()?
///         - 4f64).abs() < 1e-15);
/// # Ok(()) }
/// ```
///
/// If you want to use `?` to return errors in the second closure, its
/// error type must be known and you must help Rust with a type annotation.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
/// use root1d::{bisect, Error};
/// let f = |a: f64| bisect(|x| x*x - a, 0., a.max(1.)).root();
/// let g = |a| -> Result<_, Error<_>> { Ok(f(a)? - 2.) };
/// assert!((bisect(g, 1., 5.).root()? - 4f64).abs() < 1e-15);
/// # Ok(()) }
/// ```

pub fn bisect<T, F, R>(
    f: F, a: T, b: T) -> Bisect<T, F, T::DefaultTerminate, R::Error>
where
    T: Bisectable,
    F: FnMut(T) -> R,
    R: FloatOrError<T>
{
    Bisect::new(f, a, b)
}

new_root_finding_method!(
    bisect,
    /// Bisection algorithm (for [`Copy`] types).
    Bisect<...>,);

/// Indicate whether the values `fa` and `fb` of the function at
/// endpoints of the interval \[`a`, `b`\] have opposite signs or if
/// one of them vanishes.
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
/// `Root2` in this case).  Return an error if `fa` or `fb` is not
/// finite or `fa` * `fb` > 0.
#[inline]
fn check_sign<T, E>(a: T, b: T, fa: T, fb: T) -> Result<SignChange, Error<T, E>>
where T: Bisectable {
    use SignChange::*;
    if fa.lt0() {
        if fb.gt0() {
            Ok(NegPos)
        } else if fb.lt0() {
            Err(Error::NoSignChange { a, fa, b, fb })
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            Err(Error::NotFinite{ x: b,  fx: fb })
        }
    } else if fa.gt0() {
        if fb.lt0() {
            Ok(PosNeg)
        } else if fb.gt0() {
            Err(Error::NoSignChange { a, fa, b, fb })
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

impl<T, F, R, Term> Bisect<T, F, Term, R::Error>
where
    T: Bisectable + Copy,
    F: FnMut(T) -> R,
    R: FloatOrError<T>,
    Term: Terminate<T>
{
    /// Return `Ok(r)` where `r` is an approximate root of the
    /// function (provided that it is continuous) or `Err` indicating
    /// that the function returned a NaN value or, if
    /// [`maxiter_err`][Bisect::maxiter_err] was turned on, that the
    /// maximum number of iterations was reached.
    pub fn root(&mut self) -> Result<T, Error<T, R::Error>> {
        let mut x = self.a;
        self.root_mut(&mut x).and(Ok(x))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][Bisect::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(T,T), Error<T, R::Error>> {
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
    pub fn root_mut(
        &mut self, root: &mut T
    ) -> Result<(T,T), Error<T, R::Error>> {
        Self::check_interval_bounds(self)?;
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
        let mut fa = eval_float_result(&mut self.f, a)?;
        let mut fb = eval_float_result(&mut self.f, b)?;
        macro_rules! body {
            ($lt0: ident, $gt0: ident) => {
                for _ in 0 .. self.maxiter {
                    root.assign_mid(&a, &b);
                    if self.t.stop(&a, &b, &fa, &fb) {
                        return Ok((a, b));
                    }
                    let fx = eval_float_result(&mut self.f, *root)?;
                    if fx.$lt0() { a = *root;  fa = fx; }
                    else if fx.$gt0() { b = *root;  fb = fx; }
                    else if fx.is_finite() { return Ok((a, b)) }
                    else { return Err(Error::NotFinite{ x: *root, fx }) }
                }
            }
        }
        match check_sign(a, b, fa, fb)? {
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
pub fn bisect_mut<'a, T, F, R>(
    f: F, a: &'a T, b: &'a T
) -> BisectMut<'a, T, F, T::DefaultTerminate, R::Error>
where
    T: Bisectable,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError
{
    BisectMut::new(f, a, b)
}

/// Workspace needed to run the bisection algorithm for non-Copy types.
#[derive(Clone)]
pub struct BisectMutWorkspace<T> {
    a: T, b: T, fa: T, fb: T, fx: T,
}

impl<T: Clone> BisectMutWorkspace<T> {
    /// Create a new workspace by cloning `v`.
    pub fn new(v: &T) -> Self {
        Self {
            a: v.clone(), b: v.clone(),
            fa: v.clone(), fb: v.clone(), fx: v.clone(),
        }
    }
}

new_root_finding_method! (
    bisect_mut,
    /// Bisection algorithm (for non-[`Copy`] types).
    BisectMut<'a,...>,
    workspace, Option<&'a mut BisectMutWorkspace<T>>,
    owned_workspace, Option<BisectMutWorkspace<T>>);

impl<'a, T, F, Term, E> BisectMut<'a, T, F, Term, E>
where T: Bisectable, Term: Terminate<T> {
    /// Provide variables that will be used as workspace when running
    /// the bisection algorithm.
    #[must_use]
    pub fn work(mut self, w: &'a mut BisectMutWorkspace<T>) -> Self {
        self.workspace = Some(w);
        self
    }
}

/// Same as [`check_sign`] for non-Copy types.  In addition evaluate
/// `f` at `a` and `b` and store the result in `fa` and `fb` respectively.
#[inline]
fn check_sign_mut<T,F, R>(
    a: &T, b: &T,
    f: &mut F, fa: &mut T, fb: &mut T
) -> Result<SignChange, Error<T, R::Error>>
where
    T: Bisectable,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError
{
    use SignChange::*;
    f(fa, a);
    if fa.lt0() {
        f(fb, b);
        if fb.gt0() {
            Ok(NegPos)
        } else if fb.lt0() {
            Err(Error::NoSignChange { a: a.clone(), fa: fa.clone(),
                                      b: b.clone(), fb: fb.clone() })
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            Err(Error::NotFinite{ x: b.clone(), fx: fb.clone() })
        }
    } else if fa.gt0() {
        f(fb, b);
        if fb.lt0() {
            Ok(PosNeg)
        } else if fb.gt0() {
            Err(Error::NoSignChange { a: a.clone(), fa: fa.clone(),
                                      b: b.clone(), fb: fb.clone() })
        } else if fb.is_finite() { // f(b) = 0
            Ok(Root2)
        } else {
            Err(Error::NotFinite{ x: b.clone(), fx: fb.clone() })
        }
    } else if fa.is_finite() { // f(a) = 0
        Ok(Root1)
    } else {
        Err(Error::NotFinite{ x: a.clone(), fx: fa.clone() })
    }
}

impl<'a, T, F, R, Term> BisectMut<'a, T, F, Term, R::Error>
where
    T: Bisectable,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError,
    Term: Terminate<T>
{
    /// Set `root` to a root of the function `f` (see [`bisect_mut`]).
    /// Return the final bracket if all went well or an error to
    /// indicate that the algorithm failed (e.g., when the function
    /// `f` returned a NaN value).  Note that, if you want to use the
    /// returned bracket, you must bind `self` to ensure that it lives
    /// long enough for the references to be valid.
    ///
    /// If the [`work`][`BisectMut::work`] method was not used,
    /// internal variables are constructed by cloning `root`, thereby
    /// inheriting its precision for example.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "rug")] {
    /// use root1d::bisect_mut;
    /// use rug::{Assign, Float};
    /// let f = |y: &mut Float, x: &Float| { y.assign(x - 2); };
    /// let a0 = Float::with_val(53, 0.);
    /// let b0 = Float::with_val(53, 3.);
    /// let mut bi = bisect_mut(f, &a0, &b0); // ⇒ `a`, `b` live long enough
    /// let mut x = a0.clone();
    /// let (a, b) = bi.root_mut(&mut x)?;
    /// assert!(a <= &x && &x <= b);
    /// # } Ok(()) }
    /// ```
    pub fn root_mut(
        &mut self, root: &mut T
    ) -> Result<(&T, &T), Error<T, R::Error>> {
        // If some workspace if given, use it even if internal storage
        // is available because it may have different, say, precision
        // characteristics.
        Self::check_interval_bounds(self)?;
        let BisectMutWorkspace {a, b, fa, fb, fx} = match &mut self.workspace {
            None => {
                if self.owned_workspace.is_none() {
                    self.owned_workspace = Some(
                        BisectMutWorkspace::new(root));
                }
                self.owned_workspace.as_mut().unwrap()
            }
            Some(v) => v,
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
                        return Ok((a, b));
                    }
                    eval_unit_result(&mut self.f, fx, root)?;
                    // `swap` so as to reuse allocated memory.
                    if fx.$lt0() { swap(a, root);
                                   swap(fa, fx); }
                    else if fx.$gt0() { swap(b, root);
                                       swap(fb, fx); }
                    else if fx.is_finite() {
                        a.assign(root); // `a` lifetime linked to `self`
                        return Ok((a, a))
                    } else {
                        return Err(Error::NotFinite{ x: root.clone(),
                                                     fx: fx.clone() })
                    }
                }
            }
        }
        match check_sign_mut(a, b, &mut self.f, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0) } // f(a) < 0 < f(b)
            SignChange::PosNeg => { body!(gt0, lt0) }
            SignChange::Root1 => {
                root.assign(self.a);
                return Ok((a, a))
            }
            SignChange::Root2 => {
                root.assign(self.b);
                return Ok((b, b))
            }
        }

        if self.maxiter_err {
            Err(Error::MaxIter)
        } else {
            root.assign_mid(a, b);
            Ok((a, b))
        }
    }

    /// Return a root of the function `f` (see [`bisect_mut`]) or
    /// `Err(e)` to indicate that the function `f` returned a NaN
    /// value.
    pub fn root(&mut self) -> Result<T, Error<T, R::Error>> {
        let mut root = self.a.clone();
        self.root_mut(&mut root).and(Ok(root))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][BisectMut::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(&T, &T), Error<T, R::Error>> {
        let mut x = self.a.clone();
        self.root_mut(&mut x)
    }
}


////////////////////////////////////////////////////////////////////////
//
// Toms 748 for copy types

/// Requirements on the type `T` to be able to use [`toms748`]
/// algorithm.
pub trait OrdField:
    Bisectable
    + Copy
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
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

/// Find a root of the function `f` on the interval \[`a`, `b`\],
/// with finite bounds assuming `f(a)` and `f(b)` have opposite
/// signs and `f` is continuous using Algorithm 748 by Alefeld,
/// Potro and Shi.
///
/// Trying to compute the root when `f(a)` and `f(b)` do *not*
/// have opposite signs will return the error [`Error::NoSignChange`].
/// If the function is not continuous, root-finding methods will still
/// compute a small interval at the boundary of which `f` changes sign
/// and return a point in it; [`Toms748::bracket`] and
/// [`Toms748::root_mut`] return the small interval.  This algorithm
/// works best when the function is 4 times continuously
/// differentiable on \[`a`, `b`\] and the root is simple.
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
/// # Examples
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
/// The function we apply the Toms748 algorithm to may also return an
/// error in which case the bisection stops and returns the error
/// [`Error::Fun`].
///
/// ```
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
/// use root1d::toms748;
/// let f = |a: f64| toms748(|x| x*x - a, 0., a.max(1.)).root();
/// assert!((toms748(|a| f(a).and_then(|fa| Ok(fa - 2.)), 1., 5.).root()?
///         - 4f64).abs() < 1e-15);
/// # Ok(()) }
/// ```
///
/// If you want to use `?` to return errors in the second closure, its
/// error type must be known and you must help Rust with a type annotation.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
/// use root1d::{toms748, Error};
/// let f = |a: f64| toms748(|x| x*x - a, 0., a.max(1.)).root();
/// let g = |a| -> Result<_, Error<_>> { Ok(f(a)? - 2.) };
/// assert!((toms748(g, 1., 5.).root()? - 4f64).abs() < 1e-15);
/// # Ok(()) }
/// ```
///
/// # Reference
///
/// G. E. Alefeld, F. A. Potra, and Y. Shi, “Algorithm 748:
/// enclosing zeros of continuous functions,” ACM Trans. Math. Softw.,
/// vol. 21, no. 3, pp. 327–344, Sep. 1995, doi:
/// [10.1145/210089.210111](https://dx.doi.org/10.1145/210089.210111).
pub fn toms748<T, F, R>(
    f: F, a: T, b: T) -> Toms748<T, F, T::DefaultTerminate, R::Error>
where
    T: OrdField,
    F: FnMut(T) -> R,
    R: FloatOrError<T>
{
    Toms748::new(f, a, b)
}

new_root_finding_method!(
    toms748,
    /// [`toms748`] algorithm (for [`Copy`] types).
    Toms748<...>, );

macro_rules! bracket_sign {
    // Assume $a < $c < $b and $fa.$lt0() and $fb.$gt0()
    ($a: ident $b: ident $c: ident $d: ident,
     $fa: ident $fb: ident $fc: ident $fd: ident, $self: ident, $x: ident,
     $assign: ident, $lt0: ident, $gt0: ident) => {
        if $fc.$lt0() {
            if $self.t.stop(&$c, &$b, &$fc, &$fb) {
                $x.assign_mid(&$c, &$b);
                return Ok(($c, $b))
            }
            $assign!($d, $a);  $assign!($fd, $fa);
            $assign!($a, $c);  $assign!($fa, $fc); // `$b` and `$fb` unchanged
        } else if $fc.$gt0() {
            if $self.t.stop(&$a, &$c, &$fa, &$fc) {
                $x.assign_mid(&$a, &$c);
                return Ok(($a, $c))
            }
            $assign!($d, $b);  $assign!($fd, $fb);
            $assign!($b, $c);  $assign!($fb, $fc); // `$a` and `$fa` unchanged
        } else if $fc.is_finite() {
            $assign!(*$x, $c);
            return Ok(($c, $c))
        } else {
            return Err(Error::NotFinite{ x: $c.clone(), fx: $fc.clone() })
        }
    }
}

macro_rules! assign_copy { ($y: expr, $x: ident) => { $y = $x } }

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
                      assign_copy, $lt0, $gt0)
    }
}

macro_rules! abs_lt_neg_pos { ($x: expr, $y: expr) => { - $x < $y } }
macro_rules! abs_lt_pos_neg { ($x: expr, $y: expr) => { $x < - $y } }

impl<T, F, R, Term> Toms748<T, F, Term, R::Error>
where
    T: OrdField,
    F: FnMut(T) -> R,
    R: FloatOrError<T>,
    Term: Terminate<T>,
{
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
    /// The error [`Error::NotFinite`] (resp. [`Error::Fun`]) is
    /// returned if the function `f` produces a NaN value
    /// (resp. returns an error).  When [`maxiter_err`][Toms748::maxiter_err]
    /// is turned on, the error [`Error::MaxIter`] is returned if the
    /// maximum number of iterations is reached.
    ///
    /// The advantage of Algorithm 748 compared to other methods of
    /// root-finding (such as the Bisection or Brent's method) is that
    /// it generally requires a lot less evaluations of the function
    /// to achieve a desired precision.  This is particularly
    /// interesting if the function is costly to evaluate.
    pub fn root_mut(
        &mut self, root: &mut T
    ) -> Result<(T,T), Error<T, R::Error>> {
        Self::check_interval_bounds(self)?;
        let mut a;
        let mut b;
        if self.a <= self.b {
            a = self.a;
            b = self.b;
        } else {
            a = self.b;
            b = self.a;
        };
        let mut fa = eval_float_result(&mut self.f, a)?;
        let mut fb = eval_float_result(&mut self.f, b)?;
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
                let fc1 = eval_float_result(&mut self.f, c1)?;
                bracket_copy!(a b c1 d, fa fb fc1 fd, self, root, $lt0, $gt0);
                // 4.2.3: n = 2
                let c2 = Self::newton_quadratic::<1>(a, b, d, fa, fb, fd);
                body!(step, c2, $lt0, $gt0, $abs_lt);
                // n = 3..
                for _ in 2 .. self.maxiter {
                    // 4.2.3: (a, b, d, e) = (aₙ, bₙ, dₙ, eₙ)
                    let mut c = Self::ipzero([a, b, d, e], [fa, fb, fd, fe]);
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
                let fc = eval_float_result(&mut self.f, $c)?;
                e = d; // ẽₙ  (eₙ no longer used)
                fe = fd; // f(ẽₙ)
                // (a, b, d) = (ãₙ, b̃ₙ, d̃ₙ)
                bracket_copy!(a b $c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.5
                let mut c = Self::ipzero([a, b, d, e], [fa, fb, fd, fe]);
                if !Self::is_inside_interval(c, a, b) {
                    c = Self::newton_quadratic::<3>(a, b, d, fa, fb, fd);
                };
                // 4.2.6: (a, b, d) = (a̅ₙ, b̅ₙ, d̅ₙ)
                let fc = eval_float_result(&mut self.f, c)?;
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
                if matches!(dist.twice().partial_cmp(&len),
                    Some(Ordering::Greater) | None) { // dist is NaN ?
                        c.assign_mid(&a, &b);
                    }
                // 4.2.10 = 4.1.8: (a, b, d) = (âₙ, b̂ₙ, d̂ₙ)
                let fc = eval_float_result(&mut self.f, c)?;
                e = d; // save d̅ₙ and anticipate eₙ₊₁ = d̅ₙ
                fe = fd;
                bracket_copy!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.11 = 4.1.9
                // Nothing to do for the first case.
                if (b - a).twice() >= dist_an_bn { // μ = 1/2
                    e = d;  fe = fd; // eₙ₊₁ = d̂ₙ
                    c.assign_mid(&a, &b); // reuse `c`
                    let fmid = eval_float_result(&mut self.f, c)?;
                    bracket_copy!(a b c d, fa fb fmid fd, self, root,
                                  $lt0, $gt0);
                }
            }
        }
        match check_sign(a, b, fa, fb)? {
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
            let mut r = a - fa / fab;  // NaN if f returns infinite values
            if a < r {
                if r < b {
                    r
                } else if r >= b {
                    b - (b - a).div64()
                } else { // r is NaN
                    r.assign_mid(&a, &b);
                    r
                }
            } else if r <= a {
                a + (b - a).div64()
            } else { // r is NaN
                r.assign_mid(&a, &b);
                r
            }
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
    fn ipzero([a, b, c, d]: [T;4],
              [fa, fb, fc, fd]: [T;4]) -> T {
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
    pub fn root(&mut self) -> Result<T, Error<T, R::Error>> {
        let mut x = self.a;
        self.root_mut(&mut x).and(Ok(x))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][Toms748::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(T,T), Error<T, R::Error>> {
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

        /// Divide in place `self` by 64.
        fn div64(&mut self);

        /// Perform the negation.
        fn neg_assign(&mut self);

        /// Return `true` if `self` and `other` have the same sign and
        /// are both non-zero.
        #[inline]
        fn has_same_sign(&self, other: &Self) -> bool {
            (self.gt0() && other.gt0()) || (self.lt0() && other.lt0())
        }
    }

/// Same as [`toms748`] for non-[`Copy`] types.
pub fn toms748_mut<'a, T, F, R>(
    f: F, a: &'a T, b: &'a T
) -> Toms748Mut<'a, T, F, T::DefaultTerminate, R::Error>
where
    T: OrdFieldMut + 'a,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError
{
    Toms748Mut::new(f, a, b)
}

/// Workspace needed to run Toms 748 algorithm for non-Copy types.
#[derive(Clone)]
pub struct Toms748MutWorkspace<T> {
    a: T, b: T, c: T, d: T, e: T,
    fa: T, fb: T, fc: T, fd: T, fe: T,
    t1: T, t2: T, t3: T, t4: T, t5: T,
    dist_an_bn: T,
}

impl <T: Clone> Toms748MutWorkspace<T> {
    /// Create a new workspace by cloning `v`.
    pub fn new(v: &T) -> Self {
        Self {
            a: v.clone(), b: v.clone(), c: v.clone(), d: v.clone(),
            e: v.clone(), fa: v.clone(), fb: v.clone(), fc: v.clone(),
            fd: v.clone(), fe: v.clone(), t1: v.clone(), t2: v.clone(),
            t3: v.clone(), t4: v.clone(), t5: v.clone(),
            dist_an_bn: v.clone(),
        }
    }
}

new_root_finding_method!(
    toms748_mut,
    /// [`toms748_mut`] algorithm (for non-[`Copy`] types).
    Toms748Mut<'a,...>,
    workspace, Option<&'a mut Toms748MutWorkspace<T>>,
    owned_workspace, Option<Toms748MutWorkspace<T>>);

impl<'a, T, F, Term, E> Toms748Mut<'a, T, F, Term, E>
where
    T: OrdFieldMut,
    Term: Terminate<T>
{
    /// Provide variables that will be used as workspace when running
    /// the [`toms748_mut`] function.
    #[must_use]
    pub fn work(mut self, w: &'a mut Toms748MutWorkspace<T>) -> Self {
        self.workspace = Some(w);
        self
    }
}

macro_rules! assign_mut { ($y: expr, $x: ident) => { $y.assign($x) } }

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
                      assign_mut, $lt0, $gt0)
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

impl<'a, T, F, R, Term> Toms748Mut<'a, T, F, Term, R::Error>
where
    T: OrdFieldMut + 'a,
    F: FnMut(&mut T, &T) -> R,
    R: UnitOrError,
    Term: Terminate<T>
{
    /// Return a root of the function `f` (see [`toms748_mut`]) or
    /// `Err(e)` to indicate that the function `f` returned a NaN
    /// value.
    pub fn root(&mut self) -> Result<T, Error<T, R::Error>> {
        let mut root = self.a.clone();
        self.root_mut(&mut root).and(Ok(root))
    }

    /// Return an interval containing the root.  See
    /// [`root_mut`][Toms748Mut::root_mut] for more information.
    pub fn bracket(&mut self) -> Result<(&T, &T), Error<T, R::Error>> {
        let mut x = self.a.clone();
        self.root_mut(&mut x)
    }

    /// Set `root` to a root of the function `f` (see [`toms748_mut`]).
    /// Return the final bracket if all went well or an error to
    /// indicate that the algorithm failed (e.g., when the function
    /// `f` returned a NaN value).  Note that, if you want to use the
    /// returned bracket, you must bind `self` to ensure that it lives
    /// long enough for the references to be valid.
    ///
    /// If the [`work`][`Toms748Mut::work`] method was not used,
    /// internal variables are constructed by cloning `root`, thereby
    /// inheriting its precision for example.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "rug")] {
    /// use root1d::toms748_mut;
    /// use rug::{Assign, Float};
    /// let f = |y: &mut Float, x: &Float| { y.assign(x - 2); };
    /// let a0 = Float::with_val(53, 0.);
    /// let b0 = Float::with_val(53, 3.);
    /// let mut tm = toms748_mut(f, &a0, &b0); // ⇒ `a`, `b` live long enough
    /// let mut x = a0.clone();
    /// let (a, b) = tm.root_mut(&mut x)?;
    /// assert!(a <= &x && &x <= b);
    /// # } Ok(()) }
    /// ```
    pub fn root_mut(
        &mut self, root: &mut T
    ) -> Result<(&T, &T), Error<T, R::Error>> {
        Self::check_interval_bounds(self)?;
        let Toms748MutWorkspace {
            a, b, c, d, e,
            fa, fb, fc, fd, fe,
            t1, t2, t3, t4, t5,
            dist_an_bn} = match &mut self.workspace {
                None => {
                    if self.owned_workspace.is_none() {
                        self.owned_workspace = Some(
                            Toms748MutWorkspace::new(root));
                    }
                    self.owned_workspace.as_mut().unwrap()
                }
                Some(v) => v,
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
                    return Ok((a, b))
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
                                            [a, b, d], [fa, fb, fd]);
                body!(step, $lt0, $gt0, $abs_lt);
                // n = 3..
                for _ in 2 .. self.maxiter {
                    // 4.2.3: (a, b, d, e) = (aₙ, bₙ, dₙ, eₙ)
                    Self::ipzero(c, [t1, t2, t3, t4],
                                 [a, b, d, e], [fa, fb, fd, fe]);
                    if !Self::is_inside_interval(&c, &a, &b) {
                        Self::newton_quadratic::<1>(c, [t1, t2, t3, t4, t5],
                                                    [a, b, d], [fa, fb, fd]);
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
                Self::ipzero(c, [t1, t2, t3, t4],
                             [a, b, d, e], [fa, fb, fd, fe]);
                if !Self::is_inside_interval(c, a, b) {
                    Self::newton_quadratic::<3>(c, [t1, t2, t3, t4, t5],
                                                [a, b, d], [fa, fb, fd]);
                };
                // 4.2.6: (a, b, d) = (a̅ₙ, b̅ₙ, d̅ₙ)
                (self.f)(fc, c);
                bracket_mut!(a b c d, fa fb fc fd, self, root, $lt0, $gt0);
                // 4.2.7 = 4.1.5: c = uₙ, t1 = f(uₙ)
                debug_assert!(fa.$lt0() && fb.$gt0());
                // 4.2.8 = 4.1.6
                t2.assign(b);  *t2 -= a;  // t2 = b - a
                if $abs_lt!(fa, fb, tmp = t1) {
                    t1.assign(fa);  t1.twice();  *t1 *= t2;
                    t3.assign(fa);  *t3 -= fb;  *t1 /= t3; // t1 = c̅ₙ - uₙ ≥ 0
                    // c = c̅ₙ = uₙ - 2 f(uₙ) * (b - a) / (fb - fa)
                    c.assign(a);  *c += t1;
                    if c <= a { // `t1` absorbed by `a`
                        t1.assign(t2);  t1.div64(); // t1 = |uₙ - c̅ₙ|
                        c.assign(a);  *c += t1;
                    }
                } else {
                    t1.assign(fb);  t1.twice();  *t1 *= t2;
                    t3.assign(fb);  *t3 -= fa;  *t1 /= t3; // t1 = uₙ - c̅ₙ ≥ 0
                    c.assign(b);  *c -= t1;
                    if c >= b {
                        t1.assign(t2);  t1.div64();
                        c.assign(b);  *c -= t1;
                    }
                };
                // 4.2.9 = 4.1.7: c = ĉₙ
                t1.twice(); // t1 = 2|uₙ - c̅ₙ|
                if matches!(t1.partial_cmp(&t2),
                    Some(Ordering::Greater) | None) { // Recall t2 = b - a > 0
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
        match check_sign_mut(a, b, &mut self.f, fa, fb)? {
            SignChange::NegPos => { body!(lt0, gt0, abs_lt_neg_pos_mut); }
            SignChange::PosNeg => { body!(gt0, lt0, abs_lt_pos_neg_mut); },
            SignChange::Root1 => { root.assign(a); return Ok((a, a)) },
            SignChange::Root2 => { root.assign(b); return Ok((a, a))},
        }
        if self.maxiter_err {
            return Err(Error::MaxIter)
        }
        root.assign_mid(a, b);
        Ok((a, b))
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
        r: &mut T,
        [fab, fabd, t1, den, p]: [&mut T; 5],
        [a, b, d]: [&T; 3],
        [fa, fb, fd]: [&T; 3]
    ) {
        fab.assign(fb);  *fab -= fa; // fb - fa
        t1.assign(b);  *t1 -= a; // b - a
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
        if fabd.has_same_sign(fa) {
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
            if *a < *r {
                if *r < *b {
                    // r is OK
                } else if *r >= *b {
                    r.assign(a);  *r -= b;  r.div64();  *r += b;
                } else { // r is NaN
                    r.assign_mid(a, b);
                }
            } else if *r <= *a {
                r.assign(b);  *r -= a;  r.div64();  *r += a; // a + (b-a)/64
            } else { // r is NaN
                r.assign_mid(a, b);
            }
        }
    }

    /// Compute IP(0), the value at 0 of the inverse cubic interporation.
    #[inline]
    fn ipzero(
        r: &mut T, [t1, t2, t3, t4]: [&mut T; 4],
        [a, b, c, d]: [&T; 4],
        [fa, fb, fc, fd]: [&T; 4],
    ) {
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
         $div64: ident) => {
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
                fn div64(&mut self) { $div64!(self) }

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
    macro_rules! float_div64 { ($x: ident) => { *$x *= 0.015625 } }
    impl_rug!(Float, float_new,
              Float::with_val(53, 1e-16), // Sync with `set_rtol` below
              Float::with_val(53, 1e-12), // Sync with `set_atol` below
              float_is_finite,
              fn assign_mid(&mut self, a: &Self, b: &Self) {
                  self.assign(a);
                  *self += b;
                  *self /= 2i8;
              },
              float_is_zero,
              float_div64);

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
            if self.rtol <= 0 {
                <Float as AssignRound<f64>>::assign_round(
                    &mut self.rtol, 1e-16, Round::Nearest);
            } else {
                self.rtol.assign_round(rtol, Round::Nearest);
            }
        }
        fn set_atol(&mut self, atol: U) {
            if self.atol < 0 {
                <Float as AssignRound<f64>>::assign_round(
                    &mut self.atol, 1e-12, Round::Nearest);
            } else {
                self.atol.assign_round(atol, Round::Nearest);
            }
        }
    }

    macro_rules! rational_new { () => { Rational::new() } }
    macro_rules! rational_is_finite { ($s: ident) => { true } }
    macro_rules! rational_is_zero { ($s: ident) => {
        $s.cmp0() == Ordering::Equal
    } }
    macro_rules! rational_div64 { ($x: ident) => { *$x /= 64 } }
    impl_rug!(Rational, rational_new,
              (1, 1000_0000_0000_0000u64).into(), // Sync with set_rtol below
              (1, 1000_0000_0000_0000u64).into(), // Sync with set_atol below
              rational_is_finite,
              fn assign_mid(&mut self, a: &Self, b: &Self) {
                  self.assign(a);
                  *self += b;
                  *self /= 2u8;
              },
              rational_is_zero,
              rational_div64);

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
            if self.rtol <= 0 {
                <Rational as rug::Assign<&(_, _)>>::assign(
                    &mut self.rtol,
                    &(1, 1000_0000_0000_0000u64));
            } else {
                <Rational as rug::Assign<U>>::assign(&mut self.rtol, rtol);
            }
        }
        fn set_atol(&mut self, atol: U) {
            if self.atol < 0 {
                <Rational as rug::Assign<&(_,_)>>::assign(
                    &mut self.atol,
                    &(1, 1000_0000_0000_0000u64));
            } else {
                <Rational as rug::Assign<U>>::assign(&mut self.atol, atol);
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
    use std::{fmt::{Debug, Display},
              f64::consts::PI};
    use crate::{self as root1d, NoError};
    //use test::bench::Bencher;

    type R<T> = Result<(), root1d::Error<T, NoError>>;

    #[test]
    fn error_is_static() {
        // Compile time test.
        fn _f<T>(x: T) -> Result<(), Box<dyn std::error::Error + 'static>>
        where T: Debug + Display + 'static + Clone {
            let fx = x.clone();
            Err(Box::new(root1d::Error::<_, NoError>::NotFinite{x, fx}))
        }
    }

    #[test]
    fn bisect_decr_bounds() -> R<f64> {
        let f = |x| x * x - 2.;
        assert_approx_eq!(root1d::bisect(f, 2., 0.).maxiter_err(true).root()?,
                          2f64.sqrt(), 1e-12);
        assert_approx_eq!(root1d::bisect(f, 0., -2.).maxiter_err(true).root()?,
                          - 2f64.sqrt(), 1e-12);
        Ok(())
    }

    type Toms = root1d::Toms748::<
        f64, fn(f64) -> f64, root1d::Tol<f64>, NoError>;

    #[test]
    fn test_ipzero() {
        // y³ + 3 = x
        let r = Toms::ipzero([-5., 2., 4., 11.], [-2., -1., 1., 2.]);
        assert_eq!(r, 3.);
    }

    #[test]
    fn test_newton_quadratic() {
        let r = Toms::newton_quadratic::<0>(0.,  2., 1.,
                                             -1., 3., 0.);
        assert_eq!(r, 1.25);
        let r = Toms::newton_quadratic::<1>(0.,  2., 1.,
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

    #[test]
    fn bisect_no_sign_change() {
        let f = |x: f64| x * x - 1.;
        assert!(matches!(root1d::bisect(f, -2., 2.).root(),
                         Err(root1d::Error::NoSignChange {..})));
    }

    #[test]
    fn toms748_no_sign_change() {
        let f = |x: f64| x * x - 1.;
        assert!(matches!(root1d::toms748(f, -2., 2.).root(),
                         Err(root1d::Error::NoSignChange {..})));
    }

    #[test]
    fn bisect_f_nan() {
        let f = |x: f64| (x * x - 1.).sqrt() + x;
        assert!(matches!(
            root1d::bisect(f, -2., 2.).root(),
            Err(root1d::Error::NotFinite {x: _, fx: _})));
    }

    #[test]
    fn toms748_f_nan() {
        let f = |x: f64| (x * x - 1.).sqrt() + x;
        assert!(matches!(
            root1d::toms748(f, -2., 2.).root(),
            Err(root1d::Error::NotFinite {x: _, fx: _})));
    }

    // #[bench]
    // fn bisection_f64_speed(b: &mut Bencher) {
    //     let f = |x| x * x - 2.;
    //     b.iter(|| root1d::toms748(f, 0., 100.));
    // }

    // Some functions for which roots are difficult to compute.
    // Inspired from examples in Brent, R. (1973) Algorithms for
    // Minimization without Derivatives. Englewood Cliffs, NJ: Prentice-Hall.
    #[test]
    fn difficult() -> R<f64> {
        let delta = 1e-3;
        let a = 0.;  let b = 1.;
        let fa = -(b - a - delta) / delta * 2f64.powf(b / delta);
        let f_a_delta = 2f64.powf(a / delta);
        let root = a - delta / (f_a_delta / fa - 1.);
        let f = |x: f64| {
            if a + delta <= x && x <= b { 2f64.powf(x / delta) }
            else if x == a { fa }
            else { fa + (f_a_delta - fa) * (x - a) / delta } };
        assert_approx_eq!(root1d::toms748(f, a, b).atol(1e-20).root()?,
                          root, 1e-20);
        Ok(())
    }

    #[test]
    fn all_deriv_vanishing() -> R<f64> {
        // All derivatives vanish at 0.
        let f = |x:f64| { if x == 0. { 0. }
                          else { x * (-1. / x.abs().powf(0.1)).exp() }};
        assert_approx_eq!(root1d::toms748(f, -1., 4.).atol(1e-20).root()?,
                          0., 1e-20);
        Ok(())
    }
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

    #[test]
    fn bisect_no_sign_change() {
        let f = |y: &mut Float, x: &Float| { y.assign(x * x);  *y -= 1. };
        let a = Float::with_val(53, -2_f64);
        let b = Float::with_val(53, 2_f64);
        assert!(matches!(root1d::bisect_mut(f, &a, &b).root(),
                         Err(root1d::Error::NoSignChange {..})));
    }

    #[test]
    fn toms748_no_sign_change() {
        let f = |y: &mut Float, x: &Float| { y.assign(x * x);  *y -= 1. };
        let a = Float::with_val(53, -2_f64);
        let b = Float::with_val(53, 2_f64);
        assert!(matches!(root1d::toms748_mut(f, &a, &b).root(),
                         Err(root1d::Error::NoSignChange {..})));
    }

    #[test]
    fn bisect_f_nan() {
        let f = |y: &mut Float, x: &Float| {
            y.assign(x * x);  *y -= 1.;  y.sqrt_mut();  *y -= x;
        };
        let a = Float::with_val(53, -2_f64);
        let b = Float::with_val(53, 2_f64);
        assert!(matches!(
            root1d::bisect_mut(f, &a, &b).root(),
            Err(root1d::Error::NotFinite {x: _, fx: _})));
    }

    #[test]
    fn toms748_f_nan() {
        let f = |y: &mut Float, x: &Float| {
            y.assign(x * x);  *y -= 1.;  y.sqrt_mut();  *y -= x;
        };
        let a = Float::with_val(53, -2_f64);
        let b = Float::with_val(53, 2_f64);
        assert!(matches!(
            root1d::toms748_mut(f, &a, &b).root(),
            Err(root1d::Error::NotFinite {x: _, fx: _})));
    }
}
