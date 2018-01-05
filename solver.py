import numpy as np


def mmica(X, max_iter=200, tol=None, n_cg_iter=1, verbose=False):
    '''
    Runs the majorization-minimization ICA algorithm.


    Parameters
    ----------
    X : array-like, shape (n_sources, n_samples)
        Input mixed signals matrix, where n_sources is the number of sources
        and n_samples is the number of samples.

    max_iter : int, optional
        Maximum number of iterations to perform.

    tol : float or None, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged. The condition is
        that the norm of the gradient is below tol. If None, the algorithm
        stops after max_iter.

    n_cg_iter : int, optional
        Number of iterations to perform in the conjugate gradient. Setting it
        to 1 is highly perfered. Set it to n_sources in order to have an exact
        solver.

    verbose : bool, optional
        Prints informations about the state of the algorithm if True.

    Returns
    -------

    W : array, shape (n_sources, n_sources)
        Estimated un-mixing matrix.

    '''
    n_sources, n_samples = X.shape
    W = np.eye(n_sources)
    U = u_star(X)
    for n in range(max_iter):
        W_inv = np.linalg.inv(W)
        cg_solved = parallel_cg(X, U, W_inv.T, init=W, n_cg_iter=n_cg_iter)
        for i in range(n_sources):
            cg_solved_i = cg_solved[i]
            W[i] = cg_solved_i / np.sqrt(np.dot(cg_solved_i, W_inv[:, i]))
        Y = np.dot(W, X)
        U = u_star(Y)
        if verbose or (tol is not None):
            gradient = np.dot(U * Y, Y.T) / n_samples - np.eye(n_sources)
            gradient_norm = np.linalg.norm(gradient)
            if verbose and n % 10 == 0:
                print('iteration %d, gradient norm = %.3g'
                      % (n, gradient_norm))
            if tol is not None:
                if gradient_norm < tol:
                    break
    return W


def u_star(Y):
    '''
    Returns the star function of Y.
    '''
    return 1 / (1 + np.abs(Y))


def free_product(P, X, U):
    '''
    Computes the products A^i P[i] without computing the A^i s.
    '''
    _, T = X.shape
    Y = np.dot(P, X)
    return np.dot(U * Y, X.T) / T


def parallel_cg(X, U, Q, init=None, n_cg_iter=1):
    '''
    Uses conjugate gradient to compute solve(A^i, Q[i]) for all i.
    '''
    N, _ = X.shape
    if init is None:
        sol = np.zeros((N, N), dtype=float)
    else:
        sol = init
    residuals = free_product(sol, X, U) - Q
    P = - residuals
    for n in range(n_cg_iter):
        AP = free_product(P, X, U)
        for i in range(N):
            r = residuals[i]
            ap = AP[i]
            p = P[i]
            alpha = np.dot(r, r) / np.dot(p, ap)
            sol[i] += alpha * p
            new_r = r + alpha * ap
            beta = np.dot(new_r, new_r) / np.dot(r, r)
            P[i] = - new_r + beta * p
            residuals[i] = new_r
    return sol
