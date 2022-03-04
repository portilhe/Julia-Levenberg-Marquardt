
function f_meyer( x::AbstractVector{T}, n::Int ) where {T <: AbstractFloat}
    m = length(x)
    @assert (m == 3) "Meyer's problem is defined with m == 3"
    y = Vector{T}( undef, n )
    for i in 1:n
        ui::T = 0.45 + 0.05 * (i-1)
        y[i] = x[1] * exp( 10.0 * x[2]/ (ui+x[3]) - 13.0 )
    end
    return y
end

function Df_meyer( x::AbstractVector{T}, n::Int ) where {T <: AbstractFloat}
    m = length(x)
    @assert (m == 3) "Meyer's problem is defined with m == 3"
    Df = Matrix{T}( undef, n, m )
    for i in 1:n
        ui::T   = 0.45 + 0.05 * (i-1)
		aux1::T = ui + x[3]
		aux2::T = exp( 10.0 * x[2] / aux1 - 13.0 )

        Df[i,1] = aux2
        Df[i,2] = 10.0 * x[1] * aux2 / aux1
        Df[i,3] = -Df[i,2] * x[2] / aux1
    end
    return Df
end

function meyer_test( io::IO, alt_prob::Bool )
    x  = [ 8.85
           4.0
           2.5 ]

    hy = [ 34.780
           28.610
           23.650
           19.630
           16.370
           13.720
           11.540
           9.744
           8.261
           7.030
           6.005
           5.147
           4.427
           3.820
           3.307
           2.872 ]

	n  = length(hy)

    f_meyer_16(x_)  = f_meyer(x_, n)
    Df_meyer_16(x_) = Df_meyer(x_, n)

    if alt_prob
        opt_result = levenberg_marquardt( f_meyer_16::Function, Df_meyer_16::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_meyer_16::Function, x, hy, 1000; compute_covar_matrix=true )
		println( io, "Covariance of the fit:" )
        for i in 1:3
            for j in 1:3
                printfmt( io, " {:9.8f}", opt_result.covar[i,j] )
            end
            println( io, "" )
        end
        println( io, "" )
    end

	err = lm_check_jacobian( f_meyer_16, Df_meyer_16, x )
    for i in 1:n
        printfmtln_g( io, "gradient $(i-1), err ", err[i])
    end
    println("")

    return opt_result
end
