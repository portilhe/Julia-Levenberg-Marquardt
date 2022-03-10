
function f_helical_valley( x::AbstractVector{T} ) where {T <: AbstractFloat}
    y = Vector{T}( undef, 3 )

    if x[1] < zero(T)
        θ = atan( x[2], x[1] ) / 2π + 0.5
    elseif zero(T) < x[1]
        θ = atan( x[2], x[1]) / 2π
    else
        θ = x[2] >= zero(T) ? 0.25 : -0.25
    end

	y[1] = 10( x[3] - 10θ )
	y[2] = 10( sqrt( x[1]^2 + x[2]^2 ) - 1.0 )
	y[3] = x[3]

    return y
end

function Df_helical_valley( x::AbstractVector{T} ) where {T <: AbstractFloat}
    Df = Matrix{T}( undef, 3, 3 )

	aux::T = x[1]^2 + x[2]^2

	Df[1,1] =  50.0 * x[2] / (π * aux)
	Df[1,2] = -50.0 * x[1] / (π * aux)
	Df[1,3] =  10.0

	Df[2,1] = 10.0 * x[1] / sqrt(aux)
	Df[2,2] = 10.0 * x[2] / sqrt(aux)
	Df[2,3] = 0.0

	Df[3,1] = 0.0
	Df[3,2] = 0.0
	Df[3,3] = 1.0

    return Df
end

# Helical valley function, minimum at (1.0, 0.0, 0.0)
function helical_valley_test( ::IO, alt_prob::Bool )
    x  = [ -1.0, 0.0, 0.0 ]
    hy = [  0.0, 0.0, 0.0 ]

    if alt_prob
        opt_result = levenberg_marquardt( f_helical_valley::Function, Df_helical_valley::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_helical_valley::Function, x, hy, 1000 )
    end

    return opt_result
end
