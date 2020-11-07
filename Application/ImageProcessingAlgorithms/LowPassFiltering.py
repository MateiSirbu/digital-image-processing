from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import numpy as np
import math

@RegisterAlgorithm("Filtrul Gauss", "Filtre")
@InputDialog(sigma=float)
@OutputDialog(title="Gaussian filtering output")
def gaussianFilter(f, sigma):
    
    dim = math.ceil(sigma * 4) + (1 if (math.ceil(sigma * 4) % 2 == 0) else 0)
    h = np.zeros((dim, dim), dtype=float)
    for i in range(0, dim):
        for j in range(0, dim):
            k = -(dim//2) + i
            l = -(dim//2) + j
            h[i][j] = (1/(2 * math.pi * sigma**2)) * np.exp(-(k**2 + l**2)/(2 * sigma**2))
    h = h / np.sum(h)

    g = np.copy(f)

    if len(f.shape) == 2:                         # if image is grayscale...
        for x in range(0, g.shape[0]):            # traverse blurred image
           for y in range(0, g.shape[1]):
                processed_pixel = 0.0
                for i in range(0, dim):           # traverse mask
                    for j in range(0, dim):
                        ln = x - (dim//2) + i
                        col = y - (dim//2) + j
                        if (ln < 0):              # pixel out of bounds (ln < 0), get mirrored pixel
                            ln = -ln - 1
                        elif (ln >= f.shape[0]):  # pixel out of bounds (ln >= height), get mirrored pixel
                            ln = f.shape[0] - (ln - f.shape[0] + 1)
                        else: pass                # pixel inside bounds, do not modify index
                        if (col < 0):             # pixel out of bounds (col < 0), get mirrored pixel
                            col = -col - 1
                        elif (col >= f.shape[1]): # pixel out of bounds (col >= width), get mirrored pixel
                            col = f.shape[1] - (col - f.shape[1] + 1)
                        else: pass                # pixel inside bounds, do not modify index
                        processed_pixel = processed_pixel + (h[i][j] * f[ln][col])
                g[x][y] = int(processed_pixel)
    else:                                             # if image is color...
        for x in range(0, g.shape[0]):                # traverse blurred image
            for y in range(0, g.shape[1]):
                processed_pixel = [0.0, 0.0, 0.0]
                for channel in range(0, g.shape[2]):  # traverse pixel channels
                    for i in range(0, dim):           # traverse mask
                        for j in range(0, dim):
                            ln = x - (dim//2) + i
                            col = y - (dim//2) + j
                            if (ln < 0):              # pixel out of bounds (ln < 0), get mirrored pixel
                                ln = -ln - 1
                            elif (ln >= f.shape[0]):  # pixel out of bounds (ln >= height), get mirrored pixel
                                ln = f.shape[0] - (ln - f.shape[0] + 1)
                            else: pass                # pixel inside bounds, do not modify index
                            if (col < 0):             # pixel out of bounds (col < 0), get mirrored pixel
                                col = -col - 1
                            elif (col >= f.shape[1]): # pixel out of bounds (col >= width), get mirrored pixel
                                col = f.shape[1] - (col - f.shape[1] + 1)
                            else: pass                # pixel inside bounds, do not modify index
                            processed_pixel[channel] = processed_pixel[channel] + (h[i][j] * f[ln][col][channel])
                g[x][y] = np.asarray(processed_pixel).astype(int)

    return {
            'processedImage': g
        }