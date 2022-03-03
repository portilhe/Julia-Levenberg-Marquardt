
function f_wood( x::AbstractVector{T} ) where {T <: AbstractFloat}
    m = length(x)
    n = 6 * m ÷ 4
    y = Vector{T}( undef, n )
    for i in range(1, n; step=6)
        y[i  ] = 10.0 * (x[i+1] - x[i]^2)
        y[i+1] = 1.0 - x[i]
        y[i+2] = sqrt(90.0) * (x[i+3] - x[i+2]^2)
        y[i+3] = 1.0 - x[i+2]
        y[i+4] = sqrt(10.0) * (x[i+1]+x[i+3] - 2.0)
        y[i+5] = (x[i+1] - x[i+3]) / sqrt(10.0)
    end
    return y
end

function Df_wood( x::AbstractVector{T} ) where {T <: AbstractFloat}
    m = length(x)
    n = 6 * m ÷ 4
    Df = Matrix{T}( undef, n, m )
    for i in range(1, n; step=6)
        for j in range(1, m; step=4)
            DF[i  ,j  ] = -20.0 * x[j]
            DF[i  ,j+1] =  10.0
            DF[i  ,j+2] =   0.0
            DF[i  ,j+3] =   0.0

            DF[i+1,j  ] =  -1.0
            DF[i+1,j+1] =   0.0
            DF[i+1,j+2] =   0.0
            DF[i+1,j+3] =   0.0

            DF[i+2,j  ] =   0.0
            DF[i+2,j+1] =   0.0
            DF[i+2,j+2] =  -2*sqrt(90.0) * x[j+2]
            DF[i+2,j+3] =   sqrt(90.0)

            DF[i+3,j  ] =   0.0
            DF[i+3,j+1] =   0.0
            DF[i+3,j+2] =  -1.0
            DF[i+3,j+3] =   0.0

            DF[i+4,j  ] =   0.0
            DF[i+4,j+1] =   sqrt(10.0)
            DF[i+4,j+2] =   0.0
            DF[i+4,j+3] =   sqrt(10.0)

            DF[i+5,j  ] =   0.0
            DF[i+5,j+1] =  -1.0 / sqrt(10.0)
            DF[i+5,j+2] =   0.0
            DF[i+5,j+3] =   1.0 / sqrt(10.0)
        end
    end
    return Df
end

function wood_test( alt_prob::Bool )
    x  = [ -3.0, -1.0, -3.0, -1.0 ]
    hy = [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0  ]

    if alt_prob
        opt_result = levenberg_marquardt( f_wood::Function, Df_wood::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_wood::Function, x, hy, 1000 )
    end
    return opt_result
end
