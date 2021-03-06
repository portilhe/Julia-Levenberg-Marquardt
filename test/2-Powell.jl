
function f_powell( x::AbstractVector{T} ) where {T <: AbstractFloat}
    n = length(x)
    y = similar(x)
    for i in range(1, n; step=2)
		y[i  ] = x[i]
		y[i+1] = 10.0 * x[i] / (x[i]+0.1) + 2*x[i+1]^2
    end
    return y
end

function Df_powell( x::AbstractVector{T} ) where {T <: AbstractFloat}
    n = length(x)
    Df = Matrix{T}( undef, n, n )
    for i in range(1, n; step=2)
        for j in range(1, n; step=2)
            DF[i  ,j  ] = 1.0
            DF[i  ,j+1] = 0.0
            
            Df[i+1,j  ] = 1.0 / (x[j] + 0.1)^2
            Df[i+1,j+1] = 4.0 * x[j+1]
        end
    end
    return Df
end

# Powell's function, minimum at (0, 0)
function powell_test( ::IO, alt_prob::Bool )
    x  = [ 3.0, 1.0 ]
    hy = [ 0.0, 0.0 ]
    if alt_prob
        opt_result = levenberg_marquardt( f_powell::Function, Df_powell::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_powell::Function, x, hy, 1000 )
    end
    return opt_result
end
