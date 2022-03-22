open Printf

let () =
  let sum =ref 0. in
  for _ = 1 to 10_000 do
    for i = 2 to 49 do
      let c = float i in
      let f x = x *. x -. c in
      let r = Root1D.bisection f 0. 100. ~eps:1e-10 in
      sum := !sum +. r;
    done
  done;
  printf "[OCaml] sum: %.12f\n" !sum
