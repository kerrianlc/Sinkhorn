
import numpy as np
import cv2
import random

def uniform_circ(n):
    """
    Generate uniform data on the unit circle in 2D
    :param n: number of points to generate
    :return: data matrix of shape (n,2)
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    complexes = np.exp(1j * theta)

    coords = np.array([np.real(complexes), np.imag(complexes)]).T

    return coords

def uniform_line(x,y,n):
    """
    Generate uniform data on a line.
    :param n: number of points to generate
    :return: matrix of shape (n,d) where d is the dimension of x and y.
    """
    # Generate linearly spaced values between each corresponding pair of x and y
    result = np.linspace(0, 1, n)[:, np.newaxis] * (y - x).reshape(1, -1) + x.reshape(1, -1)

    # Reshape back to the original shape, except with `n` points along the first axis
    return result.reshape(n, *x.shape)


def get_random_points_on_black(image, num_points):
    """
    Get randomly distributed points on the black areas of an RGB image.

    Parameters:
        image (np.ndarray): Input RGB image (H, W, 3).
        num_points (int): Number of random points to sample.

    Returns:
        np.ndarray: Array of shape (num_points, 2) with randomly sampled points.
    """
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a mask for black pixels (assuming black is [0, 0, 0])
    # Adjust the threshold if necessary for nearly-black pixels
    black_mask = grayscale == 0

    # Get the coordinates of black pixels
    black_coords = np.argwhere(black_mask)  # Returns (row, col) pairs

    if len(black_coords) == 0:
        raise ValueError("No black pixels found in the image.")

    # Randomly sample points from the black pixel coordinates
    sampled_indices = random.sample(range(len(black_coords)), min(num_points, len(black_coords)))
    random_points = black_coords[sampled_indices]

    return random_points


def draw_horned_ball(num_points):
    """
    Draw data from the 2D horned ball

    :return: data on the horned ball and data on the two horn.

    """
    # path = "./img/two circles.png"
    path = "./img/moon.png"

    image = cv2.imread(path)  # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure it's RGB format


    # Get random points on the horned ball
    points = get_random_points_on_black(image, num_points) / 1000

    center_a, r_a = [0.1, 0.8], 0.4
    center_b, r_b = [0.9, 0.8], 0.4
    # center_a, r_a = [0.1, 0.8], 0.3
    # center_b, r_b = [0.3, 0.2], 0.1

    # Points on the left horn
    a = points[np.sqrt((points[:, 0] - center_a[0]) ** 2 + (points[:, 1] - center_a[1]) ** 2) < r_a, :]
    # Points on the right horn
    b = points[np.sqrt((points[:, 0] - center_b[0]) ** 2 + (points[:, 1] - center_b[1]) ** 2) < r_b, :]

    return points, a, b