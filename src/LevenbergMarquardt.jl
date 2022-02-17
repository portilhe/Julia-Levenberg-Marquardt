module JLLM

export levenberg_marquardt

using LinearAlgebra

const global SING_EPS = 1e-24

struct LMOtimization{T <: Real}
    x::AbstractVector{T}     # minimizer approximation
    y::AbstractVector{T}     # value of f(x)
    stop_error::Bool         # stopped because of an error
    niter::Int               # number of iterations
    stop::Int                # reason for stopping
    nfev::Int                # number of function evaluations
    njev::Int                # number of jacobian evaluations
    nlss::Int                # number of linear systems solved, i.e. number of attempts for reducing error
    ε0_ℓ2::T                 # ||εy||_ℓ2 at initial x
    ε_ℓ2::T                  # final value of ||εy||_ℓ2
    Jtε_ℓ∞::T                # final value of ||Jtεy||_ℓ∞
    Δx2_ℓ2::T                # final value of ||Δx||^2_ℓ2
    μ_dJtJ::T                # final value of μ / max(diag(JtJ))
    covar::AbstractMatrix{T} # covariance matrix
    # function LMOtimization{T}( x::AbstractVector{T}, y::AbstractVector{T}, stop_error::Bool, niter::Int, stop::Int,
    #                           nfev::Int, njev::Int, nlss::Int, ε0_ℓ2::T, ε_ℓ2::T, Jtε_ℓ∞::T, Δx2_ℓ2::T, μ_dJtJ::T,
    #                           covar::AbstractMatrix{T} ) where {T<:Real}
    #     return new( x, y, stop_error, niter, stop, nfev, njev, nlss, ε0_ℓ2, ε_ℓ2, Jtε_ℓ∞, Δx2_ℓ2, μ_dJtJ, covar )
    # end
end


function lm_fwd_jac_approx( f::Function,
                            J::AbstractMatrix{T},
                            x::AbstractVector{T},
                            y::AbstractVector{T},
                            δ::T ) where T <: Real
    m = length(x)
    n = length(y)
    for j in 1:m
        d::T   = max( δ, abs( 1E-04 * x[j] ) ) # force evaluation
        aux::T = x[j]
        x[j]  += d
        yy     = f(x)
        x[j]   = aux # restore

        d = 1/d # invert so that divisions can be carried out faster as multiplications
        for i in 1:n
            J[i,j] = (yy[i] - y[i]) * d
        end
    end
end


function lm_mid_jac_approx( f::Function,
                            J::AbstractMatrix{T},
                            x::AbstractVector{T},
                            δ::T ) where T <: Real
    m = length(x)
    n = length(y)
    for j in 1:m
        d::T   = max( δ, abs( 1E-04 * x[j] ) ) # force evaluation
        aux::T = x[j]
        x[j]  -= d
        yym    = f(x)
        x[j]   = aux + d
        yyp    = f(x)
        x[j]   = auz # restore

        d = 0.5/d # invert so that divisions can be carried out faster as multiplications
        for i in 1:n
            J[i,j] = (yyp[i] - yym[i]) * d
        end
    end
end


function _compute_jacobian_values( Df, x, εy )
    J       = Df(x)
    Jt      = transpose(J)
    JtJ     = Jt*J
    Jtε     = Jt*εy
    Jtε_ℓ∞  = norm(Jtε,Inf)
    diagJtJ = diag(JtJ) # save diagonal entries so that augmentation can be later canceled
    x2_ℓ2   = dot(x,x)
    return ( JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2 )
end

function levenberg_marquardt( f::Function,
                              Df::Function,
                              x::AbstractVector{T},
                              hy::AbstractVector{T},
                              maxit::Int                 = 0,
                              μ::T                       = 1e-3,
                              Jtε_stop::T                = 1e-15,
                              Δx_stop::T                 = 1e-15,
                              εy_stop::T                 = 1e-20,
                              compute_covar_matrix::Bool = false ) where T <: Real
    n = length(hy)
    m = length(x)
    if n < m
        error("levenberg_marquardt(): cannot solve a problem with fewer measurements [$n] than unknowns [$m]")
    end

    if maxit == 0
        maxit = 100*(N+1)
    end

    nfev::Int = 0
    njev::Int = 0
    nlss::Int = 0

    Δx2_ℓ2::T = convert( T, Inf )
    Jtε_ℓ∞::T = convert( T, 0.0 )
    λ         = 2

    # compute y0 = f(x0), the initial value
    y::AbstractVector{T} = f(x)
    nfev += 1
    
    # compute the L2 norm of the error εy = hy - f(x)
    εy::AbstractVector{T} = hy - y
    εy_ℓ2::T              = dot(εy,εy)
    εy0_ℓ2::T             = εy_ℓ2
    
    stop::Int = 0
    if !isfinite(εy_ℓ2)
        stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error

    elseif εy_ℓ2 <= εy_stop # error is small
        stop = 6

    else
        # Compute the Jacobian J at x,  J^t J,  J^t εy,  ||J^t εy||_∞ and ||x||^2.
        JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2 = _compute_jacobian_values( Df, x, εy )
        njev += 1

        # check for convergence 
        if Jtε_ℓ∞ <= Jtε_stop
            Δx2_ℓ2 = 0  # no increment for x in this case
            stop   = 1  # stopped by small gradient Jtε
        else
            # compute initial damping factor
            μ *= maximum( diagJtJ );
        end
    end

    niter::Int = 0
    while niter < maxit && stop == 0 # Outter loop
        niter += 1

        # determine increment using adaptive damping
        while true # inner loop
            # augment normal equations
            JtJ += μ*I

            # solve augmented equations
            issolved::Bool = false
            Δx             = similar(x)
            try
                Δx       = JtJ\Jtε
                issolved = true
                nlss    += 1
            catch
                pass
            end

            if issolved
                # compute x's new estimate and ||Δx||^2
                xΔx = x + Δx
                Δx2_ℓ2 = dot(Δx, Δx)

                if Δx2_ℓ2 <= Δx_stop^2 * x2_ℓ2 # relative change in x is small, stop
                    stop = 2  # stopped by small Δx
                    break     # from inner loop
                end

                if Δx2_ℓ2 >= (x2_ℓ2 + Δx_stop) / SING_EPS  # almost singular
                    stop = 4  # singular matrix. Should restart from current p with increased mu
                    break     # from inner loop 
                end

                # evaluate function at x + Δx
                yΔy   = f(xΔx)
                nfev += 1

                # compute ||ε(yΔy)||_2 */
                εyΔy    = hy - yΔy
                εyΔy2_ℓ2 = dot(εyΔy,εyΔy)

                # If sum of squares is not finite, most probably is due to a user error.
                # This check makes sure that the inner loop does not run indefinitely.
                if !isfinite(εyΔy2_ℓ2)
                    stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                    break    # from inner loop 
                end

                dL = dot( Δx, μ*Δx + Jtε  )
                dF = εy_ℓ2 - εyΔy2_ℓ2

                if dL > zero(T) && dF > zero(T) # reduction in error, increment is accepted
                    μ *= convert( T, max( 1/3, 1 - (2*dF/dL - 1)^3 ) )
                    λ  = 2

                    # update x's estimate
                    x  = xΔx

                    # update y,  εy and ||εy||_2
                    y     = yΔy
                    εy    = εyΔy
                    εy_ℓ2 = εyΔy2_ℓ2

                    break # from inner loop
                end
            end # if issolved

            # if this point is reached, either the linear system could not be solved or
            # the error did not reduce; in any case, the increment must be rejected

            μ *= λ
            λ <<= 1 # λ = 2*λ
            if λ < 0 # λ has wrapped around (overflown)
                stop = 5 # no further error reduction is possible. Should restart with increased λ
                break
            end

            # restore diagonal J^T J entries
            JtJ += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))

        end # inner loop

        # Note that y and εy have been updated at a previous iteration
        if εy_ℓ2 <= εy_stop # error is small
            stop = 6
        else
            # Compute the Jacobian J at x,  J^t J,  J^t εy,  ||J^t εy||_∞ and ||x||^2.
            JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2 = _compute_jacobian_values( Df, x, εy )
            njev += 1

            # check for convergence 
            if Jtε_ℓ∞ <= Jtε_stop
                Δx2_ℓ2 = 0  # no increment for x in this case
                stop   = 1  # stopped by small gradient Jtε
            end
        end

    end # outter loop #

    if niter >= maxit
        stop = 3; # stopped by maxit #
    end

    covarm = similar(JtJ)
    if compute_covar_matrix
        try
            # restore diagonal J^T J entries
            JtJ   += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))
            covarm = (εy_ℓ2 / (n-rank(C))) * pinv(JtJ)
        catch
            pass
        end
    end

    stop_error::Bool = stop == 4 || stop == 7

    return LMOtimization( x, y, stop_error, niter, stop, nfev, njev, nlss,
                          εy0_ℓ2, εy_ℓ2, Jtε_ℓ∞, Δx2_ℓ2, μ / maximum(diagJtJ), covarm )
end # levenberg_marquardt


function _approximate_jacobian_values( f, J, x, y, δ, εy, updx, updJ, λ, K, use_ffdif, nfev, njap, newJ,
                                       JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2 )
    m = length(x)
    # Compute the Jacobian J at x,  J^t J,  J^t εy,  ||J^t εy||_∞ and ||x||^2.
    if (updx && λ > 16) || updJ == K
        if use_ffdif # use forward differences
            lm_fwd_jac_approx( f, J, x, y, δ )
            njap += 1
            nfev += m
        else # use central differences
            lm_mid_jac_approx( f, J, x, δ )
            njap += 1
            nfev += 2*m
        end
        λ    = 2
        updJ = 0
        updx = 0
        newJ = true
    end

    if newJ # Jacobian has changed, recompute J^T J, J^t e, etc
        newJ = false
        # Compute the Jacobian J at x,  J^t J,  J^t εy,  ||J^t εy||_∞ and ||x||^2.
        Jt      = transpose(J)
        JtJ     = Jt*J
        Jtε     = Jt*εy
        Jtε_ℓ∞  = norm(Jtε,Inf)
        diagJtJ = diag(JtJ) # save diagonal entries so that augmentation can be later canceled
        x2_ℓ2   = dot(x,x)
    end
    return ( JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2, updx, updJ, λ, nfev, njap, newJ )
end

function levenberg_marquardt( f::Function,
                              x::AbstractVector{T},
                              hy::AbstractVector{T},
                              maxit::Int                 = 0,
                              μ::T                       = 1e-3,
                              Jtε_stop::T                = 1e-15,
                              Δx_stop::T                 = 1e-15,
                              εy_stop::T                 = 1e-20,
                              δ::T                       = 1e-6,
                              compute_covar_matrix::Bool = false ) where T <: Real
    n = length(hy)
    m = length(x)
    if n < m
        error("levenberg_marquardt(): cannot solve a problem with fewer measurements [$n] than unknowns [$m]")
    end

    if maxit == 0
        maxit = 100*(n+1)
    end

    use_ffdif::Bool = δ > 0
    δ               = abs(δ)

    Δx2_ℓ2::T = convert( T, Inf )
    Jtε_ℓ∞::T = convert( T, 0.0 )
    λ::Int    = 20  # force computation of J

    # compute y0 = f(x0), the initial value
    y::AbstractVector{T} = f(x)
    nfev::Int            = 1
    J                    = Matrix{T}( undef, n, m )
    JtJ                  = Matrix{T}( undef, n, n )
    Jtε                  = similar(x)
    diagJtJ              = diag(JtJ)
    x2_ℓ2                = dot(x,x)

    # compute the L2 norm of the error εy = hy - f(x)
    εy::AbstractVector{T} = hy - y
    εy_ℓ2::T              = dot(εy,εy)
    εy0_ℓ2::T             = εy_ℓ2

    njap       = 0
    updx::Bool = true
    updJ::Int  = 0
    newJ::Bool = false
    K::Int     = max( m, 10 )
    nlss::Int  = 0
    
    stop::Int = 0
    if !isfinite(εy_ℓ2)
        stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
    end

    niter::Int = 0
    while niter < maxit && stop == 0 # Outter loop
        niter += 1

        if εy_ℓ2 <= εy_stop # error is small
            stop = 6
            break
        end

        JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2, updx, updJ, λ, nfev, njap, newJ =
        _approximate_jacobian_values( f, J, x, y, δ, εy, updx, updJ, λ, K, use_ffdif, nfev, njap, newJ,
                                      JtJ, Jtε, Jtε_ℓ∞, diagJtJ, x2_ℓ2 )
        
        # check for convergence 
        if Jtε_ℓ∞ <= Jtε_stop
            Δx2_ℓ2 = 0  # no increment for x in this case
            stop   = 1  # stopped by small gradient Jtε
            break
        elseif niter == 1
            # compute initial damping factor
            μ *= maximum( diagJtJ );
        end
        
        # determine increment using adaptive damping

        # augment normal equations
        JtJ += μ*I

        # solve augmented equations
        Δx = similar(x)
        issolved::Bool = false
        try
            Δx       = JtJ\Jtε
            issolved = true
            nlss    += 1
        catch
            pass
        end
        
        if issolved
            # compute x's new estimate and ||Δx||^2
            xΔx = x + Δx
            Δx2_ℓ2 = dot(Δx, Δx)

            if Δx2_ℓ2 <= Δx_stop^2 * x2_ℓ2 # relative change in x is small, stop
                stop = 2  # stopped by small Δx
                break
            end

            if Δx2_ℓ2 >= (x2_ℓ2 + Δx_stop) / SING_EPS  # almost singular
                stop = 4  # singular matrix. Should restart from current p with increased mu
                break
            end

            # evaluate function at x + Δx
            yΔy   = f(xΔx)
            nfev += 1

            # compute ||ε(yΔy)||_2 */
            εyΔy    = hy - yΔy
            εyΔy2_ℓ2 = dot(εyΔy,εyΔy)

            # If sum of squares is not finite, most probably is due to a user error.
            # This check makes sure that the inner loop does not run indefinitely.
            if !isfinite(εyΔy2_ℓ2)
                stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                break
            end

            dF = εy_ℓ2 - εyΔy2_ℓ2
            dL = dot( Δx, μ*Δx + Jtε  )

            if updx || dF > 0 # update jac
                J += (yΔy - y - J*Δx) * transpose(Δx) / Δx2_ℓ2
                updJ += 1;
                newJ  = true;
            end

            if dL > zero(T) && dF > zero(T) # reduction in error, increment is accepted
                μ *= convert( T, max( 1/3, 1 - (2*dF/dL - 1)^3 ) )
                λ  = 2

                # update x's estimate
                x  = xΔx

                # update εy and ||εy||_2
                εy    = εyΔy
                y     = yΔy
                εy_ℓ2 = εyΔy2_ℓ2
                updx  = true
                continue
            end
        end # issolved

        # if this point is reached, either the linear system could not be solved or
        # the error did not reduce; in any case, the increment must be rejected

        μ *= λ
        λ <<= 1 # λ = 2*λ
        if λ < 0 # λ has wrapped around (overflown)
            stop = 5 # no further error reduction is possible. Should restart with increased λ
            break
        end

        # restore diagonal J^T J entries
        JtJ += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))

    end # Outter loop

    if niter >= maxit
        stop = 3; # stopped by maxit #
    end

    covarm = similar(JtJ)
    if compute_covar_matrix
        try
            # restore diagonal J^T J entries
            JtJ   += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))
            covarm = (εy_ℓ2 / (n-rank(C))) * pinv(JtJ)
        catch
            pass
        end
    end

    stop_error::Bool = stop == 4 || stop == 7

    return LMOtimization( x, y, stop_error, niter, stop, nfev, njap, nlss,
                          εy0_ℓ2, εy_ℓ2, Jtε_ℓ∞, Δx2_ℓ2, μ / maximum(diagJtJ), covarm )

end # function levenberg_marquardt

end # Module JLLM