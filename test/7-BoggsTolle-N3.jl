
function f_boggs_tolle_n3( x::AbstractVector{T} ) where {T <: AbstractFloat}
	t1::T = ( x[1] - x[2]       )^2
	t2::T = ( x[2] + x[3] - 2.0 )^2
	t3::T = ( x[4]        - 1.0 )^2
	t4::T = ( x[5]        - 1.0 )^2

    y = (t1 + t2 + t3 + t4) * ones( T, 5 )

    return y
end

function Df_boggs_tolle_n3( x::AbstractVector{T} ) where {T <: AbstractFloat}
	t1::T = 2.0( x[1] - x[2]       )
	t2::T = 2.0( x[2] + x[3] - 2.0 )
	t3::T = 2.0( x[4]        - 1.0 )
	t4::T = 2.0( x[5]        - 1.0 )

    Df = Matrix{T}( undef, 5, 5 )
    Df[1,1] = Df[2,1] = Df[3,1] = Df[4,1] = Df[5,1] = t1
    Df[1,2] = Df[2,2] = Df[3,2] = Df[4,2] = Df[5,2] = t2 - t1
    Df[1,3] = Df[2,3] = Df[3,3] = Df[4,3] = Df[5,3] = t2
    Df[1,4] = Df[2,4] = Df[3,4] = Df[4,4] = Df[5,4] = t3
    Df[1,5] = Df[2,5] = Df[3,5] = Df[4,5] = Df[5,5] = t4

    return Df
end

# Boggs - Tolle problem 3 (linearly constrained), minimum at (-0.76744, 0.25581, 0.62791, -0.11628, 0.25581)
# constr1: x[1] + 3*x[2]        = 0
# constr2: x[3] + x[4] - 2*x[5] = 0
# constr3: x[2] - x[5]          = 0
function boggs_tolle_n3_test( ::IO, alt_prob::Bool )
	x  = [ 2.0, 2.0, 2.0, 2.0, 2.0  ]
	hy = [ 0.0, 0.0, 0.0, 0.0, 0.0  ]

	A = [ 1.0 3.0 0.0 0.0  0.0
          0.0 0.0 1.0 1.0 -2.0
          0.0 1.0 0.0 0.0 -1.0 ]

    b = [ 0.0, 0.0, 0.0 ]

    if alt_prob
        opt_result = levenberg_marquardt_lec( f_boggs_tolle_n3::Function, Df_boggs_tolle_n3::Function, x, hy, A, b, 1000 )
    else
        opt_result = levenberg_marquardt_lec( f_boggs_tolle_n3::Function, x, hy, A, b, 1000 )
    end

    return opt_result
end
