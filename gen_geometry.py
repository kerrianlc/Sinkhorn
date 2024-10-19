
import numpy as np



def uniform_circ(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    complexes = np.exp(1j * theta)

    coords = np.array([np.real(complexes), np.imag(complexes)]).T

    return coords

def uniform_line(x,y,n):

    # Generate linearly spaced values between each corresponding pair of x and y
    result = np.linspace(0, 1, n)[:, np.newaxis] * (y - x).reshape(1, -1) + x.reshape(1, -1)

    # Reshape back to the original shape, except with `n` points along the first axis
    return result.reshape(n, *x.shape)