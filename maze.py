import argparse, cv2, numpy as np

OUTPUT_DIR = "./output/"

# Crea un parser per gli argomenti della riga di comando
parser = argparse.ArgumentParser(description='Risolutore di labirinti con Python e OpenCV.')
parser.add_argument('filename', type=str, help='Nome del file dell\'immagine del labirinto')

args = parser.parse_args()
filename = args.filename

# Estrai solo il nome del file senza estensione e percorso
f = filename.split('/')[-1].split('.')[0]

# Crea il percorso per l'immagine risolta
output_f = f"{OUTPUT_DIR}{f}_solved.png"

def print_solution(img, mask):
    # Splitting the channels of maze
    b, g, r = cv2.split(img)

    # Masking out the green and red colour from the solved path
    r = cv2.bitwise_and(r, r, mask=mask)
    b = cv2.bitwise_and(b, b, mask=mask)

    return cv2.merge((b, g, r))

def find_solution(img):
    # Binary conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inverting thresholding will give us a binary image with a white wall and a black background.
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Finding contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    dc = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5)
    dc = cv2.drawContours(dc, contours, 1, (0, 0, 0), 5)

    ret, thresh = cv2.threshold(dc, 240, 255, cv2.THRESH_BINARY)

    # Creating a kernel for morphological operations
    kernel = np.ones((15, 15), np.uint8)

    # Dilate
    dilation = cv2.dilate(thresh, kernel, iterations = 1)

    # Erosion
    erosion = cv2.erode(dilation, kernel, iterations = 1)

    # Find differences between two images
    diff = cv2.absdiff(dilation, erosion)

    mask_inv = cv2.bitwise_not(diff)

    return mask_inv

# Legge l'immagine del labirinto
maze = cv2.imread(filename)

# Risolve il labirinto
print("Solving the maze...")

res = find_solution(maze)

# Stampa la soluzione e salva l'immagine del labirinto risolto
print(f"Solved and saved the maze image in {output_f}.")

sol = print_solution(maze, res)
cv2.imwrite(output_f, sol)