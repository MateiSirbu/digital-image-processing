from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import numpy as np
import math, queue

def binarization(image, threshold):
    image[image < threshold] = 0
    image[image >= threshold] = 255
    return image

def dilate(f):
    g = np.copy(f)
    ln_end = f.shape[0] - 1
    col_end = f.shape[1] - 1
    for ln in range(0, f.shape[0]):
        for col in range(0, f.shape[1]):
            found_white_pixel = False
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if (((x == 0) and (y == 0)) or (ln + x < 0) or (col + y < 0) \
                        or (ln + x > ln_end) or (col + y > col_end)):
                            continue
                    else:
                        if (f[ln + x][col + y] == 255):
                            g[ln][col] = 255
                            found_white_pixel = True
            if (found_white_pixel == False):
                g[ln][col] = 0
    return g

def erode(f):
    g = np.copy(f)
    ln_end = f.shape[0] - 1
    col_end = f.shape[1] - 1
    for ln in range(0, f.shape[0]):
        for col in range(0, f.shape[1]):
            found_black_pixel = False
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if (((x == 0) and (y == 0)) or (ln + x < 0) or (col + y < 0) \
                        or (ln + x > ln_end) or (col + y > col_end)):
                            continue
                    else:
                        if (f[ln + x][col + y] == 0):
                            g[ln][col] = 0
                            found_black_pixel = True
            if (found_black_pixel == False):
                g[ln][col] = 255
    return g

@RegisterAlgorithm("Closing", "Morphology")
@OutputDialog(title="Closing output")
def closing(image):
    if len(image.shape) == 2:
        image = binarization(image, 128)
        image = dilate(image)
        image = erode(image)
        return {
            'processedImage': image
        }
    else:
        return {
            'outputMessage': "Error: image is not grayscale!"
        }