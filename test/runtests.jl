include("../src/LevenbergMarquardt.jl")
include("0-Rosenbrock.jl")
include("1-ModRosenbrock.jl")
include("2-Powell.jl")
include("3-Wood.jl")
include("4-Meyer.jl")

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

function printfmt_g( io, pre_s, x, pos_s=""; thresh=5, mantissa=5 )
	print(io, pre_s)
	if x == 0
		t = 1
	else
		t = Int(round(abs(log10(abs(x))),RoundUp))
	if t >= thresh
		printfmt( io, "{:9.$(mantissa)e}", x )
	else
		printfmt( io, "{:9.$(t+mantissa)f}", x )
	end
	print(io, pos_s)
end

function printfmtln_g( io, pre_s, x, pos_s=""; thresh=5, mantissa=5 )
	printfmt_g( io, pre_s, x, pos_s; thresh=5, mantissa=5 )
	println(io,"")
end

function output_results( io::IO, probname::AbstractString, lmo::JLLM.LMOtimization )
    println( io, "Results for $probname:" )
    printfmtln( io, "Levenberg-Marquardt returned after $(lmo.niter) iterations" )
    print( io, "\nSolution   x    =" )
	for xi in lmo.x
		printfmt_g( io, " ", xi )
    end
    print( io, "\nFunc value f(x) =" )
	for yi in lmo.y
		printfmt_g( io, " ", yi )
    end
    println( io, "\n\nMinimization info:" )
	printfmtln_g( io, "Initial ||e||_2       : ", lmo.ε0_ℓ2  )
	printfmtln_g( io, "Final   ||e||_2       : ", lmo.ε_ℓ2   )
	printfmtln_g( io, "Final   ||J^T e||_inf : ", lmo.Jtε_ℓ∞ )
	printfmtln_g( io, "Final   ||Δx||_2      : ", lmo.Δx2_ℓ2 )
	printfmtln_g( io, "Final   mu/max[J^T J] : ", lmo.μ_dJtJ )
	printfmtln( io, "# iterations          : {:4d}",   lmo.niter  )
	printfmtln( io, "Stopping reason {:2d}    : {:s}", lmo.stop, map_reason(lmo.stop) )
	printfmtln( io, "# function evaluations: {:4d}",   lmo.nfev   )
	printfmtln( io, "# Jacobian evaluations: {:4d}",   lmo.njev   )
	printfmtln( io, "# lin systems solved  : {:4d}",   lmo.nlss   )
    println( io )
end

problems = [  ("Rosenbrock function",            rosenbrock_test    ),
			  ("Modified Rosenbrock problem",    mod_rosenbrock_test),
			  ("Powell's function",              powell_test        ),
			  ("Wood's function",                wood_test          ),
			  ("Meyer's (reformulated) problem", meyer_test         ),
			  ("Osborne's problem",              osborne_test       ),
			  ("Helical valley function",        helical_valley_test),
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

problems = problems[1:5]

println("Running tests:")

alt_tests = false
io        = stdout

@testset "Levenberg-Marquardt" begin
    for (probname, probfunc) in problems
        println(" * $(probname)")
        lmo = probfunc( io, alt_tests )
        @test lmo.stop_error == false
        output_results( io, probname, lmo )
    end
end
