import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def conv2d(matrix, kernel):
    output = np.zeros((matrix.shape[0]-kernel.shape[0]+1, matrix.shape[1]-kernel.shape[1]+1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = np.sum(matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

if __name__ == '__main__':
    img = np.array(Image.open('len_std.jpg').convert('L')) / 255
    kern = np.array([
        [-0.5, 0, 0.5],
        [-1.0, 0, 1.0],
        [-0.5, 0, 0.5],
    ])
    conv1 = conv2d(img, kern)
    conv2 = conv2d(img, kern.T)
    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(conv1)
    plt.subplot(133)
    plt.imshow(conv2)
    plt.show()

