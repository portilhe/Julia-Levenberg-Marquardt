
function f_osborne( x::AbstractVector{T}, n::Int ) where {T <: AbstractFloat}
    m = length(x)
    @assert (m == 5) "Osborne's problem is defined with m == 5"
    y = Vector{T}( undef, n )
    for i in 1:n
        t::T = 10.0 * (i-1)
		y[i] = x[1] + x[2] * exp(-x[4]*t) + x[3] * exp(-x[5]*t);
    end
    return y
end

function Df_osborne( x::AbstractVector{T}, n::Int ) where {T <: AbstractFloat}
    m = length(x)
    @assert (m == 5) "Osborne's problem is defined with m == 5"
    Df = Matrix{T}( undef, n, m )
    for i in 1:n
		t::T = 10.0 * (i-1)
		aux1::T = exp(-x[4]*t)
		aux2::T = exp(-x[5]*t)

        Df[i,1] = one(T)
        Df[i,2] = aux1
        Df[i,3] = aux2
        Df[i,4] = -x[2] * t * aux1
        Df[i,5] = -x[3] * t * aux2
    end
    return Df
end

function osborne_test( io::IO, alt_prob::Bool )
    x  = [ 0.5
           1.5
           1.0
           0.01
           0.02 ]

    hy = [ 0.844
           0.908
           0.932
           0.936
           0.925
           0.908
           0.881
           0.85
           0.818
           0.784
           0.751
           0.718
           0.685
           0.658
           0.628
           0.603
           0.58
           0.558
           0.538
           0.522
           0.506
           0.49
           0.478
           0.467
           0.457
           0.448
           0.438
           0.431
           0.424
           0.42
           0.414
           0.411
           0.406 ]

    n  = length(hy) # n = 33

    f_osborne_33(x_)  = f_osborne(x_, n)
    Df_osborne_33(x_) = Df_osborne(x_, n)

    if alt_prob
        opt_result = levenberg_marquardt( f_osborne_33::Function, Df_osborne_33::Function, x, hy, 1000 )
    else
        opt_result = levenberg_marquardt( f_osborne_33::Function, x, hy, 1000 )
    end

    return opt_result
end
