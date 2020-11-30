from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import numpy as np
import math, queue

# the same Gaussian Filter implemented in LowPassFiltering.py
def gaussianFilter(f, sigma):
    dim = math.ceil(sigma * 4) + (1 if (math.ceil(sigma * 4) % 2 == 0) else 0)
    h = np.zeros((dim, dim), dtype=float)
    for i in range(0, dim):
        for j in range(0, dim):
            k = -(dim//2) + i
            l = -(dim//2) + j
            h[i][j] = (1/(2 * math.pi * sigma**2)) * \
                np.exp(-(k**2 + l**2)/(2 * sigma**2))
    h = h / np.sum(h)

    g = np.zeros(f.shape, dtype=np.ubyte)

    for x in range(0, g.shape[0]):
        for y in range(0, g.shape[1]):
            processed_pixel = 0.0
            for i in range(0, dim):
                for j in range(0, dim):
                    ln = x - (dim//2) + i
                    col = y - (dim//2) + j
                    if (ln < 0):
                        ln = -ln - 1
                    elif (ln >= f.shape[0]):
                        ln = f.shape[0] - (ln - f.shape[0] + 1)
                    else:
                        pass
                    if (col < 0):
                        col = -col - 1
                    elif (col >= f.shape[1]):
                        col = f.shape[1] - (col - f.shape[1] + 1)
                    else:
                        pass
                    processed_pixel = processed_pixel + (h[i][j] * f[ln][col])
            g[x][y] = np.round(processed_pixel)
    return g


def highPassFiltering(f, T1):
    # applying the Sobel operator
    gradients = np.zeros(f.shape, dtype=np.longlong)
    # "angles" are integers: 4 main directions, therefore not using radians
    angles = np.zeros(f.shape, dtype=np.ubyte)
    x_end = f.shape[0] - 1
    y_end = f.shape[1] - 1
    for x in range(0, gradients.shape[0]):
        for y in range(0, gradients.shape[1]):
            # computing edge cases indices (retrieving mirrored pixel instead of going out-of-bounds)
            xMinus1 = x if x == 0 else x - 1
            xPlus1 = x if x == x_end else x + 1
            yMinus1 = y if y == 0 else y - 1
            yPlus1 = y if y == y_end else y + 1
            # computing f_x
            f_x = int(f[xPlus1][yMinus1]) - int(f[xMinus1][yMinus1]) + 2 * int(f[xPlus1][y]) \
                - 2 * int(f[xMinus1][y]) + int(f[xPlus1]
                                               [yPlus1]) - int(f[xMinus1][yPlus1])
            # computing f_y
            f_y = int(f[xMinus1][yPlus1]) - int(f[xMinus1][yMinus1]) + 2 * int(f[x][yPlus1]) \
                - 2 * int(f[x][yMinus1]) + int(f[xPlus1]
                                               [yPlus1]) - int(f[xPlus1][yMinus1])
            # adding gradient of pixel to matrix
            gradients[x][y] = np.sqrt((f_x ** 2) + (f_y ** 2))
            # computing contour angles
            if (f_x == 0):
                angles[x][y] = 2     # vertical contour
            else:
                theta = np.arctan(f_y/f_x)
                if ((theta >= -3 * np.pi/8) and (theta < -np.pi/8)):
                    angles[x][y] = 1  # NE-heading contour
                elif ((theta >= -np.pi/8) and (theta < np.pi/8)):
                    angles[x][y] = 0  # horizontal contour
                elif ((theta >= np.pi/8) and (theta < 3 * np.pi/8)):
                    angles[x][y] = 3  # SE-heading contour
                else:
                    angles[x][y] = 2  # vertical contour
    # scaling up the gradients, then returning the results
    gradients = gradients / gradients.max() * 255
    # removing pixels below threshold T1; suppressing them during the next step is useless
    gradients[gradients <= T1] = 0
    return gradients.astype(np.ubyte), angles


def nonMaximaSuppression(g, a):
    x_end = g.shape[0] - 1
    y_end = g.shape[1] - 1
    # making contours 1 pixel wide...
    for x in range(0, g.shape[0]):
        for y in range(0, g.shape[1]):
            if (g[x][y] != 0):
                if (a[x][y] == 2):                             # vertical contour => horizontal gradient & mask
                    for y_mask in range(y - 2, y + 3):         # linear mask traversal
                        if (y_mask < 0): continue              # mask hit edge, move on: mask can't be used yet
                        elif (y_mask >= g.shape[1]): break     # mask hit edge, halt: mask ends here
                        else:                                  # mask inside picture, attempt suppression
                            if (g[x][y_mask] > g[x][y]):
                                g[x][y] = 0
                                break
                elif (a[x][y] == 0):                           # horizontal contour => vertical gradient & mask
                    for x_mask in range(x - 2, x + 3):         # linear mask traversal
                        if (x_mask < 0): continue              # mask hit edge, move on: mask can't be used yet
                        elif (x_mask >= g.shape[0]): break     # mask hit edge, halt: mask ends here
                        else:                                  # mask inside picture, attempt suppression
                            if (g[x_mask][y] > g[x][y]):
                                g[x][y] = 0
                                break
                elif (a[x][y] == 3):                           # SE-heading contour => NE-heading gradient & mask
                    for k in range(-2, 3):                     # diagonal mask traversal
                        x_mask = x - k
                        y_mask = y + k
                        if (x_mask >= g.shape[0]): continue    # mask hit edge, move on: mask can't be used yet
                        elif (y_mask < 0): continue            # mask hit edge, move on: mask can't be used yet
                        elif (x_mask < 0): break               # mask hit edge, halt: mask ends here
                        elif (y_mask >= g.shape[1]): break     # mask hit edge, halt: mask ends here
                        else:                                  # mask inside picture, attempt suppression
                            if (g[x_mask][y] > g[x][y]):
                                g[x][y] = 0
                                break
                else:                                          # NE-heading contour => SE-heading gradient & mask
                    for k in range(-2, 3):                     # diagonal mask traversal
                        x_mask = x + k
                        y_mask = y + k
                        if (x_mask < 0): continue              # mask hit edge, move on: mask can't be used yet
                        elif (y_mask < 0): continue            # mask hit edge, move on: mask can't be used yet
                        elif (x_mask >= g.shape[0]): break     # mask hit edge, halt: mask ends here
                        elif (y_mask >= g.shape[1]): break     # mask hit edge, halt: mask ends here
                        else:                                  # mask inside picture, attempt suppression
                            if (g[x_mask][y] > g[x][y]):
                                g[x][y] = 0
                                break
    # returning updated picture, with thinned contours 
    return g

def hysteresisThresholding(g, T2):
    # weak edges connected to strong ones are valid edges.
    ln_end = g.shape[0] - 1
    col_end = g.shape[1] - 1
    q = queue.Queue()
    # inserting strong edges in queue
    for ln in range(0, g.shape[0]):
        for col in range(0, g.shape[1]):
            if (g[ln][col] > T2):
                g[ln][col] = 255
                q.put((ln, col))
    # while queue is not empty, extract strong edges...
    while (q.qsize() > 0):
        point = q.get()
        ln = point[0]
        col = point[1]
        # ...and query their neighbours
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (((x == 0) and (y == 0)) or (ln + x < 0) or (ln + x > ln_end) \
                    or (col + y < 0) or (col + y > col_end)): # if hitting boundary, check next neighbor
                    continue
                else: # weak edges connected to strong ones will be considered strong and added to the queue
                    if ((g[ln + x][col + y] > 0) and (g[ln + x][col + y] < T2)):
                        q.put((ln + x, col + y))
                        g[ln + x][col + y] = 255
    # pixels that have not been altered during this process will be removed
    for ln in range(0, g.shape[0]):
        for col in range(0, g.shape[1]):
            if g[ln][col] < 255:
                g[ln][col] = 0
    return g

@RegisterAlgorithm("Canny", "Filtre")
@InputDialog(T1=int, T2=int)
@OutputDialog(title="Canny filtering output")
def canny(image, T1, T2):
    if len(image.shape) == 3:
        return {
            'outputMessage': "Error: image should be grayscale."
        }
    elif T2 < T1:
        return {
            'outputMessage': "Error: T1 should be less than T2."
        }
    elif T1 < 0:
        return { 
            'outputMessage': "Error: T1 should be greater than or equal to 0."
        }
    elif T2 > 255:
        return {
            'outputMessage': "Error: T2 should be less than or equal to 255."
        }
    else:
        print('Applying the Gaussian filter (σ = 1)...')
        image = gaussianFilter(image, 1)
        print('Done. Performing high-pass filtering: computing gradients and contour angles...')
        image, angles = highPassFiltering(image, T1)
        print('Done. Performing non-maxima suppression: thinning contours...')
        image = nonMaximaSuppression(image, angles)
        print('Done. Applying hysteresis thresholding: determining which weak edges are actual edges...')
        image = hysteresisThresholding(image, T2)
        print('Done. Displaying extracted contours.')
        return {
            'processedImage': image
        }
