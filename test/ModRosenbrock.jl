const MODROSLAM = 1E02


function f_mod_rosenbrock( x::AbstractVector{T} ) where {T <: Real}
    m = length(x)
    n = 3 * m ÷ 2
    y = Vector{T}( undef, n )
    for i in range(1, n; step=3)
        y[i  ] = 10. * (x[i+1] - x[i]^2);
        y[i+1] = 1.0 - x[1];
        y[i+2] = MODROSLAM;
    end
    return y
end

function Df_mod_rosenbrock( x::AbstractVector{T} ) where {T <: Real}
    m = length(x)
    n = 3 * m ÷ 2
    Df = Matrix{T}( undef, 2, 2 )
    for i in range(1, n; step=3)
        for j in range(1,m)
            DF[i  ,j  ] = -20.0 * x[1]
            DF[i  ,j+1] = 10.0
            Df[i+1,j  ] = -1.0
            Df[i+1,j+1] = 0.0
            Df[i+2,j  ] = 0.0
            Df[i+2,j+1] = 0.0
        end
    end
    return Df
end

function prob_1( alt_prob::Bool )
    x  = [ -1.2, 1.0 ]
    hy = [  0.0, 0.0, 0.0 ]
    if alt_prob
        opt_result = levenberg_marquardt( f_mod_rosenbrock::Function, Df_mod_rosenbrock::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_mod_rosenbrock::Function, x, hy, 1000 )
    end
    return opt_result
end
