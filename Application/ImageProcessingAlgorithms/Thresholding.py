from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import numpy as np

@RegisterAlgorithm("Binarization", "Thresholding")
@InputDialog(threshold=int)
@OutputDialog(title="Binarization Output")
def binarization(image, threshold):
    """Applies binarization based on given threshold.

    :param image:
    :param threshold:
    :return:
    """
    # if the image is grayscale
    if len(image.shape) == 2:
        image[image < threshold] = 0
        image[image >= threshold] = 255
        return {
            'processedImage': image
        }
    else:
        return {
            'outputMessage': "Error: image is not grayscale!"
        }

@RegisterAlgorithm("Double-threshold binarization", "Thresholding")
@InputDialog(t1=int, t2=int)
@OutputDialog(title="Double-threshold binarization output")
def doubleThresholdBinarization(image, t1, t2):
    if (t1 >= 1 and t1 <= 254 and t2 >= 1 and t2 <= 254 and t1 <= t2 and len(image.shape) == 2):
        image = np.where((t1 <= image) & (image <= t2), 255, image)
        image = np.where((image != 255) & ((t1 > image) | (image > t2)), 0, image)
        return {
            'processedImage': image
        }
    elif len(image.shape) != 2:
        return {
            'outputMessage': "Error: image is not grayscale"
        }
    else:
        return {
            'outputMessage': "Error: invalid thresholds"
        }