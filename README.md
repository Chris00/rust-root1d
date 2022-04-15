Find roots of functions of one variable
=======================================

This crate provides algorithms over a generic type `T` to find roots
of functions `T` → `T`.  It is readily usable for `T` being `f64` and
`f32`.  Activating the feature `rug`, one can also use it with
[`rug::Float`][] and [`rug::Rational`][].

## Usage

```rust
use root1d::toms748;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let root = toms748(|x| x*x - 2., 0., 2.).rtol(1e-10).root()?;
println!("root: {}", root);
}
```

For more information, consult the
[documentation](https://docs.rs/root1d) of the latest release.


## Highlights

- Efficient & fully generic code.
- Convenient interface with optional arguments.
- Support for non-copy types (for multi-precision numbers) minimizing
  the creation of temporary values.
- State of the art root finding algorithm ([Toms748][]).

[`rug::Float`]: https://docs.rs/rug/latest/rug/struct.Float.html
[`rug::Rational`]: https://docs.rs/rug/latest/rug/struct.Rational.html
[Toms748]: https://doi.org/10.1145/210089.210111
