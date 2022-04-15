Find roots of functions of one variable
=======================================

This crate provides algorithms over a generic type `T` to find roots
of functions `T` → `T`.  It is readily usable for `T` being `f64` and
`f32` as well as
[`rug::Float`](https://docs.rs/rug/latest/rug/struct.Float.html) and
[`rug::Rational`](https://docs.rs/rug/latest/rug/struct.Rational.html).


## Highlights

- Efficient & fully generic code.
- Convenient interface with optional arguments.
- Support for non-copy types (for multi-precision numbers) minimizing
  the creation of temporary values.
- State of the art root finding algorithm ([Toms748][]).


[Toms748]: https://doi.org/10.1145/210089.210111
