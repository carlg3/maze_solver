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

print("Image:")
print(m)
plt.imshow(m)

"""## Dilation

### Output atteso
"""

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

dilate_im_ = cv2.dilate(m, kernel, iterations=1)

print("Ex Dilated Image:")
print(dilate_im_)

"""### Output sperimentale"""

def dilation_(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applica la dilatazione morfologica su un'immagine binaria senza usare cv2.dilate.

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

dilate_im_2 = dilation_(m, kernel, iterations=1)

print("Dilated Image:")
print(dilate_im_2)

"""## Erosion

### Output atteso
"""

eroded_im_ = cv2.erode(m, kernel, iterations=1)

print("Ex Eroded Image:")
print(eroded_im_)

"""### Output sperimentale"""

def erosion_(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applica l'erosione morfologica su un'immagine binaria senza usare cv2.erode.

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

eroded_im_2 = erosion_(m, kernel, iterations=1)

print("Eroded Image:")
print(eroded_im_2)