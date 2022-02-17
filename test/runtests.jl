include("../src/LevenbergMarquardt.jl")
include("Rosenbrock.jl")
include("ModRosenbrock.jl")

using Test
using .JLLM
using Formatting

function map_reason(reason::Int)::String
    if reason == 1
        return "small gradient J^T e"
    elseif reason == 2
        return "small Dp";
    elseif reason == 3
        return "itmax";
    elseif reason == 4
        return "singular matrix. Restart from current p with increased mu";
    elseif reason == 5
        return "no further error reduction is possible. Restart with increased mu";
    elseif reason == 6
        return "small ||e||_2";
    elseif reason == 7
        return "invalid (i.e. NaN or Inf) 'func' values. This is a user error";
    else
        return "I don't know why";
    end
end

function output_results( io::IO, probname::AbstractString, lmo::JLLM.LMOtimization )
    println( io, "Results for $probname:" )
    printfmtln( io, "Levenberg-Marquardt returned after $(lmo.niter) iterations" )
    print( io, "\nSolution   x    =" )
	for xi in lmo.x
        printfmt( io, " {:9.6f}", xi )
    end
    print( io, "\nFunc value f(x) =" )
	for yi in lmo.y
        printfmt( io, " {:9.6f}", yi )
    end
    println( io, "\n\nMinimization info:" )
	printfmtln( io, "Initial ||e||_2       : {:9.6f}", lmo.ε0_ℓ2  )
	printfmtln( io, "Final   ||e||_2       : {:9.6e}", lmo.ε_ℓ2   )
	printfmtln( io, "Final   ||J^T e||_inf : {:9.6e}", lmo.Jtε_ℓ∞ )
	printfmtln( io, "Final   ||Δx||_2      : {:9.6e}", lmo.Δx2_ℓ2 )
	printfmtln( io, "Final   mu/max[J^T J] : {:9.6e}", lmo.μ_dJtJ )
	printfmtln( io, "# iterations          : {:4d}",   lmo.niter  )
	printfmtln( io, "Stopping reason {:2d}    : {:s}", lmo.stop, map_reason(lmo.stop) )
	printfmtln( io, "# function evaluations: {:4d}",   lmo.nfev   )
	printfmtln( io, "# Jacobian evaluations: {:4d}",   lmo.njev   )
	printfmtln( io, "# lin systems solved  : {:4d}",   lmo.nlss   )
    println( io )
end

probnames = [ "Rosenbrock function",
			  "modified Rosenbrock problem",
			  "Powell's function",
			  "Wood's function",
			  "Meyer's (reformulated) problem",
			  "Osborne's problem",
			  "helical valley function",
			  "Boggs & Tolle's problem #3",
			  "Hock - Schittkowski problem #28",
			  "Hock - Schittkowski problem #48",
			  "Hock - Schittkowski problem #51",
			  "Hock - Schittkowski problem #01",
			  "Hock - Schittkowski modified problem #21",
			  "hatfldb problem",
			  "hatfldc problem",
			  "equilibrium combustion problem",
			  "Hock - Schittkowski modified #1 problem #52",
			  "Schittkowski modified problem #235",
			  "Boggs & Tolle modified problem #7",
			  "Hock - Schittkowski modified #2 problem #52",
			  "Hock - Schittkowski modified problem #76" ]
probfuncs = [ prob_0, prob_1 ]

probnames = probnames[1:2]

println("Running tests:")

# @testset "Levenberg-Marquardt" begin
#     for (probname, prob_test) in zip(probnames, my_tests)
#         println(" * $(probname)")
#         include(prob_test)
#         output_results( stdout, probname, lmo )
#     end
# end

alt_tests = false

@testset "Levenberg-Marquardt" begin
    for (probname, probfunc) in zip(probnames, probfuncs)
        println(" * $(probname)")
        lmo = probfunc(alt_tests)
        @test lmo.stop_error == false
        output_results( stdout, probname, lmo )
    end
end

# lmo = prob_0(alt_tests)
# output_results( stdout, probnames[1], lmo )
