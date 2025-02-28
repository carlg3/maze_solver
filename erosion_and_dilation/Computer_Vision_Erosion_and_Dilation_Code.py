"""
## Intro
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

m = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

m = m.astype(np.uint8)

print("Matrice originale:\n")
print(m)
plt.imshow(m)


"""## Dilation

### Output atteso
"""

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

dilate_im_exp = cv2.dilate(m, kernel, iterations=1)

"""### Output sperimentale"""

def dilation_(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Immagine binaria di input.
        kernel (np.ndarray): Elemento strutturale per la dilatazione.
        iterations (int): Numero di iterazioni della dilatazione.

    Returns:
        np.ndarray: Immagine dilatata.
    """

    kernel = kernel.astype(np.uint8)

    result = image.copy()
    height, width = image.shape

    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    for _ in range(iterations):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        output = result.copy()

        for i in range(height):
            for j in range(width):
                if np.any(padded[i:i+k_height, j:j+k_width] & kernel):
                    output[i, j] = 1

        result = output

    return result

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

dilate_im_ = dilation_(m, kernel)

print("Matrice attesa post-dilate:\n")
print(dilate_im_exp)
plt.imshow(dilate_im_exp)

print("Matrice ottenuta post-dilate implementata:\n")
print(dilate_im_)
plt.imshow(dilate_im_)

"""## Erosion

### Output atteso
"""

eroded_im_exp = cv2.erode(m, kernel)

"""### Output sperimentale"""

def erosion_(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Immagine binaria di input.
        kernel (np.ndarray): Elemento strutturale per l'erosione.
        iterations (int): Numero di iterazioni dell'erosione.

    Returns:
        np.ndarray: Immagine erosa.
    """

    kernel = kernel.astype(np.uint8)
    result = image.copy()
    height, width = image.shape

    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    for _ in range(iterations):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=1)
        output = result.copy()

        for i in range(height):
            for j in range(width):
                region = padded[i:i+k_height, j:j+k_width]
                if np.array_equal(region * kernel, kernel):
                    output[i, j] = 1
                else:
                    output[i, j] = 0

        result = output

    return result

eroded_im_ = erosion_(m, kernel)

print("Matrice attesa post-erode:\n")
print(eroded_im_exp)
plt.imshow(eroded_im_exp)

print("Matrice ottenuta post-erode implementata:\n")
print(eroded_im_)
plt.imshow(eroded_im_)
