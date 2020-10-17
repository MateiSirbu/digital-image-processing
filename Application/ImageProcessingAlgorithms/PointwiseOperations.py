"""
Module docstring?
"""
import numpy as np

from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.OutputDecorators import OutputDialog

@RegisterAlgorithm("Invert", "PointwiseOp")
def invert(image):
    """Inverts every pixel of the image.

    :param image:
    :return:
    """
    return {
        'processedImage': np.invert(image)
    }


@RegisterAlgorithm("Mirror", "PointwiseOp")
def mirror(image):
    for i in range(image.shape[0]):
        image[i] = image[i][::-1]
    return {
        'processedImage': image
    }


@RegisterAlgorithm("Color Histogram Equalization", "PointwiseOp")
@OutputDialog(title="Color Histogram Equalization output")
def colorHistEq(image):
    if len(image.shape) != 3:
        return {
            'outputMessage': "Error: image is not color"
        }
    else:
        image_rgb_scaled = image / 255
        image_hsv = np.copy(image).astype(np.float)
        no_pixels = image.shape[0] * image.shape[1]

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                [r, g, b] = image_rgb_scaled[i, j]

                minim = np.min([r, g, b])
                maxim = np.max([r, g, b])
                delta = maxim - minim

                if delta == 0:
                    h = np.nan
                elif r == maxim:
                    h = 60 * (((g - b) / delta) % 6)
                elif g == maxim:
                    h = 60 * (2 + ((b - r) / delta))
                else:
                    h = 60 * (4 + ((r - g) / delta))

                if h != np.nan and h < 0:
                    h = h + 360

                if delta == 0:
                    s = 0
                else:
                    s = delta / maxim

                v = maxim

                image_hsv[i, j] = [h, s, v]

        v_histogram = np.histogram(image_hsv[:, :, 2], bins=256, range=(0, 256/255))[0]
        v_histogram_cum = np.cumsum(v_histogram/no_pixels)
        lut = np.round(((v_histogram_cum - v_histogram_cum[0])/(1 - v_histogram_cum[0])) * 255)

        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                [h, s, v] = image_hsv[i, j]
                v = lut[int(v * 255)]
                c = v * s
                if not np.isnan(h):
                    h = h / 60
                    x = c * (1 - np.abs((h % 2) - 1))
                    if (0 <= h) and (h < 1):
                        [r, g, b] = [c, x, 0]
                    elif (1 <= h) and (h < 2):
                        [r, g, b] = [x, c, 0]
                    elif (2 <= h) and (h < 3):
                        [r, g, b] = [0, c, x]
                    elif (3 <= h) and (h < 4):
                        [r, g, b] = [0, x, c]
                    elif (4 <= h) and (h < 5):
                        [r, g, b] = [x, 0, c]
                    else:
                        [r, g, b] = [c, 0, x]
                else:
                    [r, g, b] = [0, 0, 0]
                m = v - c
                [r, g, b] = [r + m, g + m, b + m]
                image[i, j] = [r, g, b]
        return {
            'processedImage': image
        }
