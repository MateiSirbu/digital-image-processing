from Application.Utils.AlgorithmDecorators import RegisterAlgorithm
from Application.Utils.InputDecorators import InputDialog
from Application.Utils.OutputDecorators import OutputDialog
import numpy as np
import math

def interpolate(t, f_1, f0, f1, f2):
    A = np.array([1, t, t**2, t**3])
    B = np.array([[0,  2, 0,  0], [-1, 0,  1, 0],
                   [2, -5, 4, -1], [-1, 3, -3, 1]])
    C = np.array([f_1, f0, f1, f2])
    result = 1/2 * (A @ B @ C)
    result = np.clip(result, 0, 255)
    return result

@RegisterAlgorithm("Scalare cu interpolare bicubică", "Transformări Geometrice")
@InputDialog(scale=float)
@OutputDialog(title="Scaling and bicubic interpolation output")
def bicubic(f, scale):
    f = f.astype(int)
    if scale <= 0:
        return {
            'outputMessage': "Error: the scaling factor should be greater than zero."
        }
    if (len(f.shape) == 3):
        n, m, _ = np.round(np.array(f.shape, dtype=float) * scale).astype(int)
        g = np.zeros((n, m, f.shape[2]), dtype=np.ubyte)
    else:
        n, m = np.round(np.array(f.shape, dtype=float) * scale).astype(int)
        g = np.zeros((n, m), dtype=np.ubyte)
    
    for y in range(0, n):
        for x in range(0, m):
            yc = y / scale
            xc = x / scale
            y0 = np.floor(yc).astype(int)
            x0 = np.floor(xc).astype(int)
            ty = yc - y0
            tx = xc - x0
    
            yo_1 = np.clip(y0 - 1, 0, f.shape[0] - 1)
            yo1  = np.clip(y0 + 1, 0, f.shape[0] - 1)
            yo2  = np.clip(y0 + 2, 0, f.shape[0] - 1)
            xo_1 = np.clip(x0 - 1, 0, f.shape[1] - 1)
            xo1  = np.clip(x0 + 1, 0, f.shape[1] - 1)
            xo2  = np.clip(x0 + 2, 0, f.shape[1] - 1)
    
            # values of interpolated pixels along the X axis
            b_1 = interpolate(tx, f[yo_1,xo_1], f[yo_1,x0], f[yo_1,xo1], f[yo_1,xo2])
            b0 =  interpolate(tx, f[y0,xo_1], f[y0,x0], f[y0,xo1], f[y0,xo2])
            b1 =  interpolate(tx, f[yo1,xo_1], f[yo1,x0], f[yo1,xo1], f[yo1,xo2])
            b2 =  interpolate(tx, f[yo2,xo_1], f[yo2,x0], f[yo2,xo1], f[yo2,xo2])
            
            # value of interpolated values of interpolated pixels
            g[y, x] = np.round(interpolate(ty, b_1, b0, b1, b2))

    return {
        'processedImage': g.astype(np.ubyte)
    }
