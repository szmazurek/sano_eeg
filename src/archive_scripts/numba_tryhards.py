import numba
from numba import (
    njit,
    jit,
    vectorize,
    float32,
    float64,
    complex128,
    int16,
    complex64,
    int64,
    guvectorize,
    prange,
)
from scipy import signal
from scipy.fft import fft, ifft
import numpy as np
import time


@njit(cache=True, fastmath=True)
def jit_fft(x, N=None, axis=-1):
    return fft(x)


@njit(cache=True, fastmath=True)
def jit_ifft(x, N=None, axis=-1):
    return ifft(x)


@njit(cache=True, fastmath=True)
def hilbert_jitted(x, N: int = None, axis=-1):
    x = np.asarray(x, dtype=float32)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    Xf = jit_fft(x, N, axis=axis)
    h = np.empty_like(x)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    # if x.ndim > 1:
    #     ind = [np.newaxis] * x.ndim
    #     ind[axis] = slice(None)
    #     h = h[tuple(ind)]

    x = jit_ifft(Xf * h, axis=axis)
    return x


@njit()
def _idxiter(n, triu=True, include_diag=False):
    pos = -1
    for i in range(n):
        for j in range(i * int(triu), n):
            if not include_diag and i == j:
                continue
            else:
                pos += 1
                yield pos, i, j


@njit(cache=True, fastmath=True)
def compute_phase_lock_val_jitted(data, include_diag=False):
    """Phase Locking Value (PLV).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.


    """
    n_channels, n_times = data.shape
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    plv = np.empty((n_coefs,))
    for s, i, j in _idxiter(n_channels, include_diag=include_diag):
        if i == j:
            plv[j] = 1
        else:
            xa = hilbert_jitted(data[i, :])
            ya = hilbert_jitted(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv


def compute_phase_lock_val(data, include_diag=False):
    """Phase Locking Value (PLV).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.

    Notes
    -----
    Alias of the feature function: **phase_lock_val**. See [1]_.

    References
    ----------
    .. [1] http://www.gatsby.ucl.ac.uk/~vincenta/kaggle/report.pdf
    """
    n_channels, n_times = data.shape
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    plv = np.empty((n_coefs,))
    for s, i, j in _idxiter(n_channels, include_diag=include_diag):
        if i == j:
            plv[j] = 1
        else:
            xa = signal.hilbert(data[i, :])
            ya = signal.hilbert(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv


def compute_plv_matrix_jitted(graph):
    """Compute connectivity matrix via usage of PLV from MNE implementation.
    Args:
        graph: (np.ndarray) Single graph with shape [nodes,features] where features represent consecutive time samples and nodes represent
    electrodes in EEG.
    Returns:
        plv_matrix: (np.ndarray) PLV matrix of the input graph.
    """
    plv_conn_vector = compute_phase_lock_val_jitted(graph)

    n = int(np.sqrt(2 * len(plv_conn_vector))) + 1

    # Reshape the flattened array into a square matrix
    upper_triangular = np.zeros((n, n))
    upper_triangular[np.triu_indices(n, k=1)] = plv_conn_vector

    # Create an empty matrix for the complete symmetric matrix
    symmetric_matrix = np.zeros((n, n))

    # Fill the upper triangular part (including the diagonal)
    symmetric_matrix[np.triu_indices(n)] = upper_triangular[np.triu_indices(n)]

    # Fill the lower triangular part by mirroring the upper triangular
    plv_matrix = (
        symmetric_matrix + symmetric_matrix.T - np.diag(np.diag(symmetric_matrix))
    )

    # Add 1 to the diagonal elements
    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix


def compute_plv_matrix(graph: np.ndarray) -> np.ndarray:
    """Compute connectivity matrix via usage of PLV from MNE implementation.
    Args:
        graph: (np.ndarray) Single graph with shape [nodes,features] where features represent consecutive time samples and nodes represent
    electrodes in EEG.
    Returns:
        plv_matrix: (np.ndarray) PLV matrix of the input graph.
    """
    plv_conn_vector = compute_phase_lock_val(graph)

    n = int(np.sqrt(2 * len(plv_conn_vector))) + 1

    # Reshape the flattened array into a square matrix
    upper_triangular = np.zeros((n, n))
    upper_triangular[np.triu_indices(n, k=1)] = plv_conn_vector

    # Create an empty matrix for the complete symmetric matrix
    symmetric_matrix = np.zeros((n, n))

    # Fill the upper triangular part (including the diagonal)
    symmetric_matrix[np.triu_indices(n)] = upper_triangular[np.triu_indices(n)]

    # Fill the lower triangular part by mirroring the upper triangular
    plv_matrix = (
        symmetric_matrix + symmetric_matrix.T - np.diag(np.diag(symmetric_matrix))
    )

    # Add 1 to the diagonal elements
    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix


random_graph = np.random.rand(18, 256 * 10)
start = time.time()
plv = compute_plv_matrix_jitted(random_graph)
for i in range(10):
    start = time.time()
    plv = compute_plv_matrix_jitted(random_graph)
    print(f"Second run: {time.time() - start}")
print("##############################################")
for i in range(10):
    start = time.time()
    plv = compute_plv_matrix(random_graph)
    print(f"Third run: {time.time() - start}")
