module JLLM

export levenberg_marquardt, levenberg_marquardt_lec, lm_check_jacobian

using LinearAlgebra

const global SING_EPS = 1e-24


function lm_check_jacobian( f::Function, Df::Function, x::AbstractVector{T} ) where T <: AbstractFloat
    #  Check the Jacobian of a n-valued nonlinear function in m variables
    #  evaluated at a point p, for consistency with the function itself.
    # 
    #  Based on fortran77 subroutine CHKDER by
    #  Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
    #  Argonne National Laboratory. MINPACK project. March 1980.
    # 
    # 
    #  f points to a function from R^m --> R^n: Given an x in R^m it yields y in R^n
    #  Df points to a function implementing the Jacobian of f, whose correctness is to
    #      be tested. Given an x in R^m, Df computes into the nxm matrix j the Jacobian
    #      of f at x. Note that row i of j corresponds to the gradient of the i-th
    #      component of f, evaluated at x.
    #  x is an input array of length m containing the point of evaluation.
    #  n is the number of functions
    #
    #
    #  The returned value, err, is an array of length n containing measures
    #      of correctness of the respective gradients. If there is
    #      no severe loss of significance, then if err[i] is 1.0 the
    #      i-th gradient is correct, while if err[i] is 0.0 the i-th
    #      gradient is incorrect. For values of err between 0.0 and 1.0,
    #      the categorization is less certain. In general, a value of
    #      err[i] greater than 0.5 indicates that the i-th gradient is
    #      probably correct, while a value of err[i] less than 0.5
    #      indicates that the i-th gradient is probably incorrect.
    # 
    # 
    #  The function does not perform reliably if cancellation or
    #  rounding errors cause a severe loss of significance in the
    #  evaluation of a function. therefore, none of the components
    #  of x should be unusually small (in particular, zero) or any
    #  other value which may cause loss of significance.
    # 
    y = f(x)
    J = Df(x)
    m = length(x)
    n = length(y)
    mach_?? = eps(T)
    sqrt_?? = sqrt(mach_??)

    xx = similar(x)
    for j in 1:m
        xx[j] = x[j] == 0 ? sqrt_?? : x[j] + sqrt_??*abs(x[j])
    end

    yy = f(xx)

    ?? = zeros(T, n)
    for j in 1:m
        aux = x[j] == 0 ? one(T) : abs(x[j])
        for i in 1:n
            ??[i] += aux * J[i,j]
        end
    end

    mach_100?? = 100 * mach_??
    log10_??   = log10(sqrt_??)
    for i in 1:n
        aux = one(T)
        if y[i] != zero(T) && yy[i] != zero(T) && abs(yy[i]-y[i]) >= mach_100?? * abs(y[i])
            aux = sqrt_?? * abs( (yy[i]-y[i])/sqrt_?? - ??[i] ) / ( abs(yy[i]) + abs(y[i]) )
        end

        if aux >= sqrt_??
            ??[i] = zero(T)
        elseif aux > mach_??
            ??[i] = (log10(aux) - log10_??) / log10_??
        else
            ??[i] = one(T)
        end
    end

    return ??
end # function lm_check_jacobian


struct LMOtimization{T <: AbstractFloat}
    x::AbstractVector{T}     # minimizer approximation
    y::AbstractVector{T}     # value of f(x)
    stop_error::Bool         # stopped because of an error
    niter::Int               # number of iterations
    stop::Int                # reason for stopping
    nfev::Int                # number of function evaluations
    njev::Int                # number of jacobian evaluations
    nlss::Int                # number of linear systems solved, i.e. number of attempts for reducing error
    ??0_???2::T                 # ||??y||_???2 at initial x
    ??_???2::T                  # final value of ||??y||_???2
    Jt??_??????::T                # final value of ||Jt??y||_??????
    ??x2_???2::T                # final value of ||??x||^2_???2
    ??_dJtJ::T                # final value of ?? / max(diag(JtJ))
    covar::AbstractMatrix{T} # covariance matrix
    # function LMOtimization{T}( x::AbstractVector{T}, y::AbstractVector{T}, stop_error::Bool, niter::Int, stop::Int,
    #                           nfev::Int, njev::Int, nlss::Int, ??0_???2::T, ??_???2::T, Jt??_??????::T, ??x2_???2::T, ??_dJtJ::T,
    #                           covar::AbstractMatrix{T} ) where {T<:AbstractFloat}
    #     return new( x, y, stop_error, niter, stop, nfev, njev, nlss, ??0_???2, ??_???2, Jt??_??????, ??x2_???2, ??_dJtJ, covar )
    # end
end


function lm_fwd_jac_approx( f::Function,
                            J::AbstractMatrix{T},
                            x::AbstractVector{T},
                            y::AbstractVector{T},
                            ??::T ) where T <: AbstractFloat
    m = length(x)
    n = length(y)
    for j in 1:m
        d::T   = max( ??, abs( 1E-04 * x[j] ) ) # force evaluation
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
                            ??::T ) where T <: AbstractFloat
    m = length(x)
    n = length(y)
    for j in 1:m
        d::T   = max( ??, abs( 1E-04 * x[j] ) ) # force evaluation
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


function _compute_jacobian_values( Df, x, ??y )
    J       = Df(x)
    Jt      = transpose(J)
    JtJ     = Jt*J
    Jt??     = Jt*??y
    Jt??_??????  = norm(Jt??,Inf)
    diagJtJ = diag(JtJ) # save diagonal entries so that augmentation can be later canceled
    x2_???2   = dot(x,x)
    return ( JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2 )
end

function levenberg_marquardt( f::Function,
                              Df::Function,
                              x::AbstractVector{T},
                              hy::AbstractVector{T},
                              maxit::Int                 = 0,
                              ??::T                       = 1e-3,
                              Jt??_stop::T                = 1e-15,
                              ??x_stop::T                 = 1e-15,
                              ??y_stop::T                 = 1e-20,
                              compute_covar_matrix::Bool = false ) where T <: AbstractFloat
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

    ??x2_???2::T = convert( T, Inf )
    Jt??_??????::T = convert( T, 0.0 )
    ??         = 2

    # compute y0 = f(x0), the initial value
    y::AbstractVector{T} = f(x)
    nfev += 1
    
    # compute the L2 norm of the error ??y = hy - f(x)
    ??y::AbstractVector{T} = hy - y
    ??y_???2::T              = dot(??y,??y)
    ??y0_???2::T             = ??y_???2
    
    stop::Int = 0
    if !isfinite(??y_???2)
        stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error

    elseif ??y_???2 <= ??y_stop # error is small
        stop = 6

    else
        # Compute the Jacobian J at x,  J^t J,  J^t ??y,  ||J^t ??y||_??? and ||x||^2.
        JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2 = _compute_jacobian_values( Df, x, ??y )
        njev += 1

        # check for convergence 
        if Jt??_?????? <= Jt??_stop
            ??x2_???2 = 0  # no increment for x in this case
            stop   = 1  # stopped by small gradient Jt??
        else
            # compute initial damping factor
            ?? *= maximum( diagJtJ )
        end
    end

    niter::Int = -1
    while niter < maxit && stop == 0 # Outter loop
        niter += 1

        # determine increment using adaptive damping
        while true # inner loop
            # augment normal equations
            JtJ += ??*I

            # solve augmented equations
            issolved::Bool = false
            ??x             = similar(x)
            try
                ??x       = JtJ\Jt??
                issolved = true
                nlss    += 1
            catch
                pass
            end

            if issolved
                # compute x's new estimate and ||??x||^2
                x??x = x + ??x
                ??x2_???2 = dot(??x, ??x)

                if ??x2_???2 <= ??x_stop^2 * x2_???2 # relative change in x is small, stop
                    stop = 2  # stopped by small ??x
                    break     # from inner loop
                end

                if ??x2_???2 >= (x2_???2 + ??x_stop) / SING_EPS  # almost singular
                    stop = 4  # singular matrix. Should restart from current p with increased mu
                    break     # from inner loop 
                end

                # evaluate function at x + ??x
                y??y   = f(x??x)
                nfev += 1

                # compute ||??(y??y)||_2 */
                ??y??y    = hy - y??y
                ??y??y2_???2 = dot(??y??y,??y??y)

                # If sum of squares is not finite, most probably is due to a user error.
                # This check makes sure that the inner loop does not run indefinitely.
                if !isfinite(??y??y2_???2)
                    stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                    break    # from inner loop 
                end

                dL = dot( ??x, ??*??x + Jt??  )
                dF = ??y_???2 - ??y??y2_???2

                if dL > zero(T) && dF > zero(T) # reduction in error, increment is accepted
                    ?? *= convert( T, max( 1/3, 1 - (2*dF/dL - 1)^3 ) )
                    ??  = 2

                    # update x's estimate
                    x  = x??x

                    # update y,  ??y and ||??y||_2
                    y     = y??y
                    ??y    = ??y??y
                    ??y_???2 = ??y??y2_???2

                    break # from inner loop
                end
            end # if issolved

            # if this point is reached, either the linear system could not be solved or
            # the error did not reduce; in any case, the increment must be rejected

            ?? *= ??
            ?? <<= 1 # ?? = 2*??
            if ?? < 0 # ?? has wrapped around (overflown)
                stop = 5 # no further error reduction is possible. Should restart with increased ??
                break
            end

            # restore diagonal J^T J entries
            JtJ += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))

        end # inner loop

        # Note that y and ??y have been updated at a previous iteration
        if ??y_???2 <= ??y_stop # error is small
            stop = 6
        else
            # Compute the Jacobian J at x,  J^t J,  J^t ??y,  ||J^t ??y||_??? and ||x||^2.
            JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2 = _compute_jacobian_values( Df, x, ??y )
            njev += 1

            # check for convergence 
            if Jt??_?????? <= Jt??_stop
                ??x2_???2 = 0  # no increment for x in this case
                stop   = 1  # stopped by small gradient Jt??
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
            covarm = (??y_???2 / (n-rank(C))) * pinv(JtJ)
        catch
            pass
        end
    end

    stop_error::Bool = stop == 4 || stop == 7

    return LMOtimization( x, y, stop_error, niter, stop, nfev, njev, nlss,
                          ??y0_???2, ??y_???2, Jt??_??????, ??x2_???2, ?? / maximum(diagJtJ), covarm )
end # levenberg_marquardt (with Jacobian)


function _approximate_jacobian_values( f, J, x, y, ??, ??y, updx, updJ, ??, K, use_ffdif, nfev, njap, newJ,
                                       JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2 )
    m = length(x)
    # Compute the Jacobian J at x,  J^t J,  J^t ??y,  ||J^t ??y||_??? and ||x||^2.
    if (updx && ?? > 16) || updJ == K
        if use_ffdif # use forward differences
            lm_fwd_jac_approx( f, J, x, y, ?? )
            njap += 1
            nfev += m
        else # use central differences
            lm_mid_jac_approx( f, J, x, ?? )
            njap += 1
            nfev += 2*m
        end
        ??    = 2
        updJ = 0
        updx = 0
        newJ = true
    end

    if newJ # Jacobian has changed, recompute J^T J, J^t e, etc
        newJ = false
        # Compute the Jacobian J at x,  J^t J,  J^t ??y,  ||J^t ??y||_??? and ||x||^2.
        Jt      = transpose(J)
        JtJ     = Jt*J
        Jt??     = Jt*??y
        Jt??_??????  = norm(Jt??,Inf)
        diagJtJ = diag(JtJ) # save diagonal entries so that augmentation can be later canceled
        x2_???2   = dot(x,x)
    end
    return ( JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2, updx, updJ, ??, nfev, njap, newJ )
end

function levenberg_marquardt( f::Function,
                              x::AbstractVector{T},
                              hy::AbstractVector{T},
                              maxit::Int                 = 0,
                              ??::T                       = 1e-3,
                              Jt??_stop::T                = 1e-15,
                              ??x_stop::T                 = 1e-15,
                              ??y_stop::T                 = 1e-20,
                              ??::T                       = 1e-6;
                              compute_covar_matrix::Bool = false ) where T <: AbstractFloat
    n = length(hy)
    m = length(x)
    if n < m
        error("levenberg_marquardt(): cannot solve a problem with fewer measurements [$n] than unknowns [$m]")
    end

    if maxit == 0
        maxit = 100*(n+1)
    end

    use_ffdif::Bool = ?? > 0
    ??               = abs(??)

    ??x2_???2::T = convert( T, Inf )
    Jt??_??????::T = convert( T, 0.0 )
    ??::Int    = 20  # force computation of J

    # compute y0 = f(x0), the initial value
    y::AbstractVector{T} = f(x)
    nfev::Int            = 1
    J                    = Matrix{T}( undef, n, m )
    JtJ                  = Matrix{T}( undef, n, n )
    Jt??                  = similar(x)
    diagJtJ              = diag(JtJ)
    x2_???2                = dot(x,x)

    # compute the L2 norm of the error ??y = hy - f(x)
    ??y::AbstractVector{T} = hy - y
    ??y_???2::T              = dot(??y,??y)
    ??y0_???2::T             = ??y_???2

    njap       = 0
    updx::Bool = true
    updJ::Int  = 0
    newJ::Bool = false
    K::Int     = max( m, 10 )
    nlss::Int  = 0
    
    stop::Int = 0
    if !isfinite(??y_???2)
        stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
    end

    niter::Int = -1
    while niter < maxit && stop == 0 # Outter loop
        niter += 1

        if ??y_???2 <= ??y_stop # error is small
            stop = 6
            break
        end

        JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2, updx, updJ, ??, nfev, njap, newJ =
        _approximate_jacobian_values( f, J, x, y, ??, ??y, updx, updJ, ??, K, use_ffdif, nfev, njap, newJ,
                                      JtJ, Jt??, Jt??_??????, diagJtJ, x2_???2 )
        
        # check for convergence 
        if Jt??_?????? <= Jt??_stop
            ??x2_???2 = 0  # no increment for x in this case
            stop   = 1  # stopped by small gradient Jt??
            break
        elseif niter == 0
            # compute initial damping factor
            ?? *= maximum( diagJtJ )
        end
        
        # determine increment using adaptive damping

        # augment normal equations
        JtJ += ??*I

        # solve augmented equations
        ??x = similar(x)
        issolved::Bool = false
        try
            ??x       = JtJ\Jt??
            issolved = true
            nlss    += 1
        catch e
            issolved = false
        end
        
        if issolved
            # compute x's new estimate and ||??x||^2
            x??x = x + ??x
            ??x2_???2 = dot(??x, ??x)

            if ??x2_???2 <= ??x_stop^2 * x2_???2 # relative change in x is small, stop
                stop = 2  # stopped by small ??x
                break
            end

            if ??x2_???2 >= (x2_???2 + ??x_stop) / SING_EPS  # almost singular
                stop = 4  # singular matrix. Should restart from current p with increased mu
                break
            end

            # evaluate function at x + ??x
            y??y   = f(x??x)
            nfev += 1

            # compute ||??(y??y)||_2 */
            ??y??y    = hy - y??y
            ??y??y2_???2 = dot(??y??y,??y??y)

            # If sum of squares is not finite, most probably is due to a user error.
            # This check makes sure that the inner loop does not run indefinitely.
            if !isfinite(??y??y2_???2)
                stop = 7 # stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                break
            end

            dF = ??y_???2 - ??y??y2_???2
            dL = dot( ??x, ??*??x + Jt??  )

            if updx || dF > 0 # update jac
                J += (y??y - y - J*??x) * transpose(??x) / ??x2_???2
                updJ += 1
                newJ  = true
            end

            if dL > zero(T) && dF > zero(T) # reduction in error, increment is accepted
                ?? *= convert( T, max( 1/3, 1 - (2*dF/dL - 1)^3 ) )
                ??  = 2

                # update x's estimate
                x  = x??x

                # update ??y and ||??y||_2
                ??y    = ??y??y
                y     = y??y
                ??y_???2 = ??y??y2_???2
                updx  = true
                continue
            end
        end # issolved

        # if this point is reached, either the linear system could not be solved or
        # the error did not reduce; in any case, the increment must be rejected

        ?? *= ??
        ?? <<= 1 # ?? = 2*??
        if ?? < 0 # ?? has wrapped around (overflown)
            stop = 5 # no further error reduction is possible. Should restart with increased ??
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
            JtJ    += diagm(size(JtJ)..., diagJtJ) - diagm(size(JtJ)..., diag(JtJ))
            covarm  = pinv(JtJ)
            covarm *= (??y_???2 / (n-rank(covarm)))
        catch
            pass
        end
    end

    stop_error::Bool = stop == 4 || stop == 7

    return LMOtimization( x, y, stop_error, niter, stop, nfev, njap, nlss,
                          ??y0_???2, ??y_???2, Jt??_??????, ??x2_???2, ?? / maximum(diagJtJ), covarm )

end # function levenberg_marquardt (without Jacobian)


function lm_lec_elim( A::AbstractMatrix{T}, b::AbstractVector{T} ) where T <: AbstractFloat
    # This function implements an elimination strategy for linearly constrained
    # optimization problems. The strategy relies on QR decomposition to transform
    # an optimization problem constrained by Ax=b to an equivalent, unconstrained
    # one. Also referred to as "null space" or "reduced Hessian" method.
    # See pp. 430-433 (chap. 15) of "Numerical Optimization" by Nocedal-Wright
    # for details.
    #
    # A is mxn with m <= n and rank(A) = m
    # Two matrices Y and Z of dimensions nxm and nx(n-m) are computed from A^T so that
    # their columns are orthonormal and every x can be written as x = Y*b + Z*x_z =
    # c + Z*x_z, where c = Y*b is a fixed vector of dimension n and x_z is an
    # arbitrary vector of dimension n-m. Then, the problem of minimizing f(x)
    # subject to Ax = b is equivalent to minimizing f(c + Z*x_z) with no constraints.
    # The computed Y and Z are such that any solution of Ax = b can be written as
    # x = Y*x_y + Z*x_z for some x_y, x_z. Furthermore, A*Y is nonsingular, A*Z = 0
    # and Z spans the null space of A.
    #
    # The function accepts A, b and computes c, Y, Z.
    m, n = size(A)
    if m > n
        error("Matrix of constraints cannot have more rows [$m] than columns [$n] in levenberg_marquardt_lec_elim()!")
    end

    Aqr = LinearAlgebra.qr( transpose(A), Val(true) )

    aux::T = max( 1e-12, n * 10.0 * eps(T) * abs(Aqr.R[1]) ) # threshold. n is max(m, n); ensure that threshold is not too small
    rank   = 0
    for i in 1:m
        # loop across R's diagonal elements
        if abs(Aqr.R[i,i]) > aux 
            rank += 1
        else
            # diagonal is arranged in absolute decreasing order
            break
        end
    end

    if rank < m
        error( "Constraints matrix in levenberg_marquardt_lec_elim() is not of full row rank (i.e. $(rank) < $(n))!"
               * " Make sure that you do not specify redundant or inconsistent constraints." )
    end

    Y = (Aqr.Q[:,1:rank])*inv(transpose(Aqr.R)) * Aqr.P
    c = Y*b
    Z = Aqr.Q[:,rank+1:end]

    return c, Z
end # function lm_lec_elim

struct LMLecData{T <: AbstractFloat}
    c::Vector{T}
    Z::Matrix{T}
    x::Vector{T}
    f::Function
    Df::Function
end

function lmlec_func( xx::AbstractVector{T}; lec_data::LMLecData ) where T <: AbstractFloat
    # constrained measurements: given xx, compute the measurements at c + Z*xx */

    # x = c + Z*xx
    mul!( lec_data.x, lec_data.Z, xx )
    for i in 1:length(lec_data.x)
        lec_data.x[i] += lec_data.c[i]
    end

    return lec_data.f(lec_data.x)
end

function lmlec_jacf( xx::AbstractVector{T}; lec_data::LMLecData ) where T <: AbstractFloat
    # constrained Jacobian: given xx, compute the Jacobian at c + Z*xx
    # Using the chain rule, the Jacobian with respect to xx equals the
    # product of the Jacobian with respect to x (at c + Z*x) times Z

    # x = c + Z*xx
    mul!( lec_data.x, lec_data.Z, xx )
    lec_data.x += lec_data.c

	# return the Jacobian J*Z
    return lec_data.Df(lec_data.x) * Z
end

function levenberg_marquardt_lec( f::Function,
                                  Df::Function,
                                  x::AbstractVector{T},
                                  hy::AbstractVector{T},
                                  A::AbstractMatrix{T},
                                  b::AbstractVector{T},
                                  maxit::Int                 = 0,
                                  ??::T                       = 1e-3,
                                  Jt??_stop::T                = 1e-15,
                                  ??x_stop::T                 = 1e-15,
                                  ??y_stop::T                 = 1e-20,
                                  compute_covar_matrix::Bool = false ) where T <: AbstractFloat
    # This function is similar to levenberg_marquardt except that the minimization
    # is performed subject to the linear constraints Ax=b, A is kxm, b kx1
    n    = length(hy)
    k, m = size(A)
    if length(x) != m
        error("levenberg_marquardt_lec(): Wrong dimensions of constraints matrix A [$(k)x$(m)], incompatible with dimension of x [$(length(x))]")
    end

    if n + k < m
        error("levenberg_marquardt_lec(): cannot solve a problem with fewer measurements + equality constraints [$n + $k] than unknowns [$m]")
    end

    c, Z = lm_lec_elim( A, b )

    xx = transpose(Z)*(x-c)
    if max(abs.(c + Z*xx - x0)) > 1e-3
        println("Warning: starting point not feasible in levenberg_marquardt_lec()! [$x reset to $(c+Z*xx)]")
    end

    lec_data   = LMLecData( c, Z, x , f, Df )
    f_lec(x_)  = lmlec_func( x_; lec_data=lec_data )
    Df_lec(x_) = lmlec_jacf( x_; lec_data=lec_data )
    lmo        = levenberg_marquardt( f_lec, Df_lec, x, hy, maxit, ??, Jt??_stop, ??x_stop, ??y_stop )

    x = c + Z*lmo.x
    y = f(x)

    covar = lmo.covar
	if compute_covar_matrix
        J  = Matrix{T}( undef, n, m )
        # compute the Jacobian with finite differences and use it to estimate the covariance
        lm_fwd_jac_approx( f, J, x, y, ?? )
        try
            covar  = pinv(transpose(J)*J)
            covar *= (??y_???2 / (n-rank(covar)))
        catch
            pass
        end
    end

    return LMOtimization( x, y,
                          lmo.stop_error,
                          lmo.niter,
                          lmo.stop,
                          lmo.nfev,
                          lmo.njev,
                          lmo.nlss,
                          lmo.??0_???2,
                          lmo.??_???2,
                          lmo.Jt??_??????,
                          lmo.??x2_???2,
                          lmo.??_dJtJ,
                          covar )
end # function levenberg_marquardt_lec (without jacobian)
    
    
function levenberg_marquardt_lec( f::Function,
                                  x::AbstractVector{T},
                                  hy::AbstractVector{T},
                                  A::AbstractMatrix{T},
                                  b::AbstractVector{T},
                                  maxit::Int                 = 0,
                                  ??::T                       = 1e-3,
                                  Jt??_stop::T                = 1e-15,
                                  ??x_stop::T                 = 1e-15,
                                  ??y_stop::T                 = 1e-20;
                                  compute_covar_matrix::Bool = false ) where T <: AbstractFloat
    # This function is similar to levenberg_marquardt except that the minimization
    # is performed subject to the linear constraints Ax=b, A is kxm, b kx1
    n    = length(hy)
    k, m = size(A)
    if length(x) != m
        error("levenberg_marquardt_lec(): Wrong dimensions of constraints matrix A [$(k)x$(m)], incompatible with dimension of x [$(length(x))]")
    end

    if n + k < m
        error("levenberg_marquardt_lec(): cannot solve a problem with fewer measurements + equality constraints [$n + $k] than unknowns [$m]")
    end

    c, Z = lm_lec_elim( A, b )

    xx = transpose(Z)*(x-c)
    if maximum(abs.(c + Z*xx - x)) > 1e-3
        println("Warning: starting point not feasible in levenberg_marquardt_lec()! [$x reset to $(c+Z*xx)]")
    end

    f_lec(x_) = lmlec_func( x_; lec_data=LMLecData( c, Z, x , f, f ) )
    lmo       = levenberg_marquardt( f_lec, xx, hy, maxit, ??, Jt??_stop, ??x_stop, ??y_stop )

    x = c + Z*lmo.x
    y = f(x)

    covar = lmo.covar
	if compute_covar_matrix
        J  = Matrix{T}( undef, n, m )
        # compute the Jacobian with finite differences and use it to estimate the covariance
        lm_fwd_jac_approx( f, J, x, y, ?? )
        try
            covar  = pinv(transpose(J)*J)
            covar *= (??y_???2 / (n-rank(covar)))
        catch
            pass
        end
    end

    return LMOtimization( x, y,
                          lmo.stop_error,
                          lmo.niter,
                          lmo.stop,
                          lmo.nfev,
                          lmo.njev,
                          lmo.nlss,
                          lmo.??0_???2,
                          lmo.??_???2,
                          lmo.Jt??_??????,
                          lmo.??x2_???2,
                          lmo.??_dJtJ,
                          covar )
end # function levenberg_marquardt_lec (with Jacobian)

end # Module JLLM
