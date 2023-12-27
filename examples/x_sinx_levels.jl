using Pkg
Pkg.add("NonlinearSolve")
using NonlinearSolve

N = 100_000
levels = 1.5 * rand(N)
out = zeros(N)

f(x, lv) = x * sin(x) - lv

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(NonlinearProblem{false}(
                           NonlinearFunction{false}(f), u0, levels[i]),
                           NewtonRaphson()).u
    end
end

@time f(out, levels, 1.0)

# Julia 1.9.4 (Intel Core i7-7500U CPU @ 2.70GHz)
# 0.133063 seconds (1.30 M allocations: 141.891 MiB, 14.63% gc time)
#
# Note that, on the first run, the following was reported:
# 299.023886 seconds (777.52 M allocations: 164.623 GiB, 1.72% gc time, 99.94% compilation time)
