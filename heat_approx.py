import typing as T
import numpy as np
from scipy.special import ive

try:
    import torch
except ImportError:
    torch = None


def compute_chebychev_coeff_all_torch(eigval, t, K):
    with torch.no_grad():
        eigval = eigval.detach().cpu()
        out = 2.0 * ive(torch.arange(0, K + 1), -t * eigval)
    return out


def compute_chebychev_coeff_all(eigval, t, K):
    return 2.0 * ive(
        np.arange(
            0,
            K + 1,
        ),
        -t * eigval,
    )


def expm_multiply(
    L: np.ndarray,
    X: np.ndarray,
    coeff: np.ndarray,
    eigval: np.ndarray,
):
    """Matrix exponential with Chebyshev polynomial approximation."""

    def body(carry, c):
        T0, T1, Y = carry
        T2 = (2.0 / eigval) * (L @ T1) - 2.0 * T1 - T0
        Y = Y + c * T2
        return (T1, T2, Y)

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = (1.0 / eigval) * (L @ X) - T0
    Y = Y + coeff[1] * T1

    initial_state = (T0, T1, Y)
    for c in coeff[2:]:
        initial_state = body(initial_state, c)

    _, _, Y = initial_state

    return Y


import numpy as np
from scipy.special import ive


def get_tchebyshev_polynomials(L: np.ndarray, X: np.ndarray, K: int) -> np.ndarray:
    """
    Compute Chebyshev polynomials up to degree K for the Laplacian matrix L.

    Parameters:
    L : np.ndarray
        Laplacian matrix (n x n).
    X : np.ndarray
        Identity matrix or input matrix (n x m), serves as the initial polynomial T0.
    K : int
        Number of terms in the Chebyshev expansion.

    Returns:
    np.ndarray
        Chebyshev polynomials T0, T1, ..., TK evaluated at X, stacked along the first dimension.
    """
    # Assertions to verify input sizes
    n, m = X.shape
    assert L.shape == (n, n), "Laplacian matrix L must be square (n x n)."

    # Initialize the first two Chebyshev polynomials
    T0 = X  # T0 is the identity or the input matrix X
    T1 = L @ X  # T1 = L * X

    # Store the polynomials
    chebyshev_polynomials = [T0, T1]

    # Recurrence relation for higher-degree polynomials: T_k+1 = 2L * T_k - T_k-1
    for k in range(2, K + 1):
        T2 = 2 * L @ T1 - T0
        chebyshev_polynomials.append(T2)
        T0, T1 = T1, T2  # Shift for the next iteration

    return np.stack(chebyshev_polynomials)


def get_bessel_coefficients(t: float, eigval_max: float, K: int) -> np.ndarray:
    """
    Compute the first kind modified Bessel function coefficients for the heat kernel expansion.

    Parameters:
    t : float
        Time parameter in the heat kernel.
    eigval_max : float
        Maximum eigenvalue (lambda_max) of the Laplacian.
    K : int
        Number of terms in the Chebyshev expansion.

    Returns:
    np.ndarray
        Bessel function coefficients evaluated at -t * lambda_max.
    """
    assert eigval_max > 0, "Eigenvalue maximum (eigval_max) must be positive."

    # Evaluate the coefficients at -t * lambda_max
    scaled_time = -t * eigval_max
    bessel_coefficients = 2.0 * ive(np.arange(0, K + 1), scaled_time)

    return bessel_coefficients


def expm_multiply_with_bessel(
        L: np.ndarray,
        X: np.ndarray,
        eigval_max: float,
        t: float,
        K: int
) -> np.ndarray:
    """
    Compute matrix exponential approximation using Chebyshev polynomials and modified Bessel function.

    Parameters:
    L : np.ndarray
        Laplacian matrix (n x n).
    X : np.ndarray
        Input matrix or identity matrix (n x m).
    eigval_max : float
        Maximum eigenvalue of the Laplacian (lambda_max).
    t : float
        Time parameter in the heat kernel.
    K : int
        Number of terms in the Chebyshev expansion.

    Returns:
    np.ndarray
        Approximated result of exp(-tL) @ X.
    """
    # Ensure L is square and X has compatible dimensions
    assert L.shape[0] == L.shape[1], "Laplacian matrix L must be square."
    assert L.shape[0] == X.shape[0], "Input matrix X must have the same number of rows as L."

    # Compute Chebyshev polynomials up to degree K
    chebyshev_polynomials = get_tchebyshev_polynomials(L, X, K)

    # Compute the Bessel function coefficients
    bessel_coefficients = get_bessel_coefficients(t, eigval_max, K)

    # Matrix exponential approximation using Chebyshev expansion
    Y = 0.5 * bessel_coefficients[0] * chebyshev_polynomials[0]  # First term

    # Add contributions of higher-order terms
    for k in range(1, K + 1):
        Y += bessel_coefficients[k] * chebyshev_polynomials[k]

    return Y
