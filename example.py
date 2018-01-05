import numpy as np
from solver import mmica


def whitening(Y, mode='sph'):
    '''
    Whitens the data Y using sphering or pca
    '''
    R = np.dot(Y, Y.T) / Y.shape[1]
    U, D, _ = np.linalg.svd(R)
    if mode == 'pca':
        W = U.T / np.sqrt(D)[:, None]
        Z = np.dot(W, Y)
    elif mode == 'sph':
        W = np.dot(U, U.T / np.sqrt(D)[:, None])
        Z = np.dot(W, Y)
    return Z, W


print('''

Majorization-minimization ICA example!

''')
# Fix a seed
rng = np.random.RandomState(3)

# Generate some super-Gaussian sources :
print('Generating mixed signals...')
n_sources, n_samples = 3, 10000
S = rng.laplace(size=(n_sources, n_samples))

# Mix the signals :
A = rng.randn(n_sources, n_sources)
X = np.dot(A, S)

# Whiten the observed signals :
print('Whitening the signals...')
X_white, W_white = whitening(X)

# Apply MM-ICA:
print('Running MM-ICA...')
W = mmica(X_white, max_iter=101, verbose=True)

print('Done!')

# Check that the mixing matrix is recovered :
print('''The product of the estimated unmixing matrix and the true mixing
matrix is : ''')
print(np.dot(W, np.dot(W_white, A)))
