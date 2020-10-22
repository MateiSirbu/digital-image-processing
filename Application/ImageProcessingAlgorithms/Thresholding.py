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
        image = np.where((image != 255) & (
            (t1 > image) | (image > t2)), 0, image)
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


@RegisterAlgorithm("Binarizare Otsu", "Thresholding")
@OutputDialog(title="Otsu thresholding output")
def otsu(image):
    if len(image.shape) == 2:
        no_pixels = image.shape[0] * image.shape[1]
        h = (np.histogram(image, bins=range(257), range=(-1, 255))[0])/no_pixels
        p1 = h[1] + h[0]
        p2 = 1 - p1
        sum1 = h[1]
        sum2 = np.sum([(k * h[k]) for k in range(2, 256)])
        mu1 = 0 if p1 == 0 else sum1/p1
        mu2 = 0 if p2 == 0 else sum2/p2
        betw_var_max = p1 * p2 * (mu1 - mu2) ** 2
        thr = 1
        for t in range(2, 256):
            p1 = p1 + h[t]
            p2 = 1 - p1
            sum1 = sum1 + t * h[t]
            sum2 = sum2 - t * h[t]
            mu1 = 0 if p1 == 0 else sum1/p1
            mu2 = 0 if p2 == 0 else sum2/p2
            if ((p1 * p2 * (mu1 - mu2) ** 2) >= betw_var_max):
                betw_var_max = p1 * p2 * (mu1 - mu2) ** 2
                thr = t
        binarized_image = binarization._func.__wrapped__(image=image, threshold=thr)
        return binarized_image
    else:
        return {
            'outputMessage': "Error: image is not grayscale"
        }
