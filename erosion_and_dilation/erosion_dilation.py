# Carlo Galante, mat.535663

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

def dilate_exp(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    k_center_x, k_center_y  = k_height // 2, k_width // 2
    
    # Coordinate con valori attivi nel kernel
    kernel_indices = np.where(kernel == 1)

    # Copia dell'immagine per il risultato
    output = np.zeros_like(image)
    
    # Padding dell'immagine per gestire i bordi, si aggiunge un bordo di k_center_x e k_center_y
    padded_image = np.pad(image, ((k_center_x, k_center_x), (k_center_y, k_center_y)), mode='constant', constant_values=0)
    
    # Coordinate dei pixel attivi nel kernel
    kernel_indices = np.where(kernel == 1)
    
    # Iterate over each pixel
    for x in range(height):
        for y in range(width):
            # Estrae una finestra centrata sul pixel corrente
            window = padded_image[x:x + k_height, y:y + k_width]
            
            # Applica il kernel e prende solo i valori che combaciano con il kernel
            values = window[kernel_indices]
            
            # Calcola il massimo
            if len(values) > 0:
                output[x, y] = np.max(values)
            else:
                output[x, y] = image[x, y]
    
    return output

def erode_exp(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    k_center_x, k_center_y  = k_height // 2, k_width // 2
    
    # Coordinate con valori attivi nel kernel
    kernel_indices = np.where(kernel == 1)

    # Copia dell'immagine per il risultato
    output = np.zeros_like(image)
    
    # Padding dell'immagine per gestire i bordi, si aggiunge un bordo di k_center_x e k_center_y
    padded_image = np.pad(image, ((k_center_x, k_center_x), (k_center_y, k_center_y)), mode='constant', constant_values=0)
    
    # Coordinate dei pixel attivi nel kernel
    kernel_indices = np.where(kernel == 1)
    
    # Iterate over each pixel
    for x in range(height):
        for y in range(width):
            # Estrae una finestra centrata sul pixel corrente
            window = padded_image[x:x + k_height, y:y + k_width]
            
            # Applica il kernel e prende solo i valori che combaciano con il kernel
            values = window[kernel_indices]
            
            # Calcola il minimo
            if len(values) > 0:
                output[x, y] = np.min(values)
            else:
                output[x, y] = image[x, y]
    
    return output


# Esempio di immagine binaria
m = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# m = cv2.imread('/content/download.jpg', cv2.IMREAD_GRAYSCALE)

# Elemento strutturante 3x3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Applicare la dilatazione
dilated_image = dilate_exp(m, kernel)

# Applicare l'erosione
eroded_image = erode_exp(m, kernel)

# Visualizza l'immagine originale e le immagini elaborate
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, (1, 2))
plt.title('Immagine originale')
plt.imshow(m)
plt.axis('on')

plt.subplot(2, 2, 3)
plt.title('Immagine dilated')
plt.imshow(dilated_image)
plt.axis('on')

plt.subplot(2, 2, 4)
plt.title('Immagine eroded')
plt.imshow(eroded_image)
plt.axis('on')

plt.tight_layout()
plt.show()