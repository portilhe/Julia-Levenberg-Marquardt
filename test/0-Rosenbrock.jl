const ROSD = 105.0


function f_rosenbrock( x::AbstractVector{T} ) where {T <: AbstractFloat}
    y = similar(x)
    y[1] = y[2] = (1 - x[1])^2 + ROSD * (x[2] - x[1]^2)^2
    return y
end

function Df_rosenbrock( x::AbstractVector{T} ) where {T <: AbstractFloat}
    Df = Matrix{T}( undef, 2, 2 )
    Df[1,1] = Df[2,1] = -2 + 2*x[1] - 4*ROSD*(x[2] - x[1]^2)*x[1]
    Df[1,2] = Df[2,2] = 2*ROSD*(x[2]-x[1]^2)
    return Df
end

function rosenbrock_test( alt_prob::Bool )
    x  = [ -1.2, 1.0 ]
    hy = [  0.0, 0.0 ]
    if alt_prob
        opt_result = levenberg_marquardt( f_rosenbrock::Function, Df_rosenbrock::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_rosenbrock::Function, x, hy, 1000 )
    end
    return opt_result
end
