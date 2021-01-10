from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.ndimage as ndimage

def plotMatrix(hough, diag):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(hough.astype('uint8'), cmap=cm.gray, extent=(-90, 90, -diag, diag), aspect=180/(2 * diag), vmin=0, vmax=255)
    ax.set_title('Matricea Hough rapidă')
    ax.set_xlabel("theta")
    ax.set_ylabel("raza")
    plt.show()

@RegisterAlgorithm("Fast Hough Transform", "Segmentation")
@InputDialog(sobel_threshold=int, hough_threshold=int)
@OutputDialog(title="Fast Hough Transform Output")
def fastHough(f, sobel_threshold, hough_threshold=int):
    if len(f.shape) == 3:
        return {
            'outputMessage': "Error: image should be grayscale."
        }
    if (sobel_threshold < 0 or sobel_threshold > 255 or hough_threshold < 0 or hough_threshold > 255):
        return {
            'outputMessage': "Error: thresholds should be greater than 0 and less than 255."
        }
    print('Computing Hough matrix...')
    n = f.shape[0]
    m = f.shape[1]
    y_end = f.shape[0] - 1
    x_end = f.shape[1] - 1
    diag = np.floor(np.sqrt(n ** 2 + m ** 2)).astype(int)
    hough = np.zeros((2 * diag, 181), dtype=int)
    for y in range(0, n):
        for x in range(0, m):
            # computing edge cases indices (retrieving mirrored pixel instead of going out-of-bounds)
            xMinus1 = x if x == 0 else x - 1
            xPlus1 = x if x == x_end else x + 1
            # note that the Y axis is inverted:
            yPlus1 = y if y == 0 else y - 1
            yMinus1 = y if y == y_end else y + 1
            # computing f_x
            f_x = int(f[yMinus1][xPlus1]) - int(f[yMinus1][xMinus1]) + 2 * int(f[y][xPlus1]) \
                - 2 * int(f[y][xMinus1]) + int(f[yPlus1]
                                               [xPlus1]) - int(f[yPlus1][xMinus1])
            # computing f_y
            f_y = int(f[yPlus1][xMinus1]) - int(f[yMinus1][xMinus1]) + 2 * int(f[yPlus1][x]) \
                - 2 * int(f[yMinus1][x]) + int(f[yPlus1]
                                               [xPlus1]) - int(f[yMinus1][xPlus1])
            # computing pixel gradient
            gradient = np.sqrt((f_x ** 2) + (f_y ** 2))
            if (gradient >= sobel_threshold):
                if (f_x == 0):
                    theta = 0
                else:
                    theta = np.arctan(f_y / f_x)
                r = np.round(x * np.cos(theta) + (y_end - y)
                             * np.sin(theta)).astype(int)
                # incrementing Hough matrix
                theta_deg = np.round(np.degrees(theta)).astype(int)
                hough[diag - r, theta_deg +
                      90] = hough[diag - r, theta_deg + 90] + 1

    # scaling and displaying the Hough matrix w/ Matplotlib
    print('Plotting Hough matrix...')
    hough = np.round(hough / np.max(hough) * 255).astype(np.ubyte)
    plotMatrix(hough, diag)

    # extracting the local maxima using scipy
    hough_filtered = ndimage.maximum_filter(hough, size=(5,5))
    labels, no_labels = ndimage.label(hough_filtered > hough_threshold)
    coords = ndimage.measurements.center_of_mass(hough, labels=labels, index=np.arange(1, no_labels + 1))

    # iterating through the Hough matrix and drawing contour lines
    print('Drawing contour lines...')

    # for every x in image, compute y and draw point
    for coord in coords:
        ln = coord[0]; col = coord[1]
        theta = np.radians(col - 90); r = diag - ln
        cosine = np.cos(theta); sine = np.sin(theta)
        for x in range(0, m):
            y = np.round((r - x * cosine)/sine).astype(int)
            if (y < 0):
                continue
            elif (y > y_end):
                break
            else:
                f[y_end - y][np.floor(x).astype(int)] = 255

    # for every y in image, compute x and draw point
    for coord in coords:
        ln = coord[0]; col = coord[1]
        theta = np.radians(col - 90); r = diag - ln
        cosine = np.cos(theta); sine = np.sin(theta)
        for y in range(0, n):
            x = np.round((r - y * sine)/cosine).astype(int)
            if (x < 0):
                continue
            elif (x > x_end):
                break
            else:
                f[y_end - y][np.floor(x).astype(int)] = 255

    print('Done.')
    return {
        'processedImage': f.astype(np.ubyte)
    }
