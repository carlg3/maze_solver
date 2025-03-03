# Carlo Galante, mat. 535663

import argparse, cv2, numpy as np

# Crea un parser per gli argomenti della riga di comando
parser = argparse.ArgumentParser(description='Risolutore di labirinti con Python e OpenCV.')
parser.add_argument('filename', type=str, help='Nome del file dell\'immagine del labirinto')

args = parser.parse_args()
filename = args.filename

def filecheck(input_dir="input", output_dir="output"):
    def decorator(func):
        def wrapper(filename):
            f = filename.rsplit('.', 1)[0]

            input_f = f"./{input_dir}/{filename}"
            output_f = f"./{output_dir}/{f}_solved.png"

            try:
                img = cv2.imread(input_f)

                if img is None:
                    raise FileNotFoundError(f"Errore: impossibile aprire il file {input_f}")
            except FileNotFoundError as e:
                print(e)
                exit(1)

            return func(img, output_f)
        return wrapper
    return decorator

# ----------------- Preprocessing the image -----------------
def preprocess_image(img):
    # Binary conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # [Thresh#1] Apply thresholding to invert the binary image, resulting in a white wall and a black background.
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Finding contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Drawing the contours on the image to get better walls of the maze 
    dc = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5)
    dc = cv2.drawContours(dc, contours, 1, (0, 0, 0), 5)

    # [Thresh#2] Thresholding the image to get a binary image
    ret, thresh = cv2.threshold(dc, 240, 255, cv2.THRESH_BINARY)

    return thresh

# ----------------- Solving the maze -----------------
def solve_maze(thresh):
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

# ----------------- Printing the solution -----------------
def merge_sol(img, mask):
    # Splitting the channels of maze
    b, g, r = cv2.split(img)

    # Masking out the green and red colour from the solved path
    r = cv2.bitwise_and(r, r, mask=mask)
    b = cv2.bitwise_and(b, b, mask=mask)

    return cv2.merge((b, g, r))

# ----------------- Main function -----------------
@filecheck()
def main(img, output_f):
    print("Solving maze...") 

    thresh = preprocess_image(img)
    mask = solve_maze(thresh)
    res = merge_sol(img, mask)

    print(f"Maze solved!\nSaving to '{output_f}'...")
    cv2.imwrite(output_f, res)

main(filename)
