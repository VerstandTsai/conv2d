import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def conv2d(matrix, kernel):
    output = np.zeros((matrix.shape[0]-kernel.shape[0]+1, matrix.shape[1]-kernel.shape[1]+1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = np.sum(matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

def pool(matrix, window_size):
    pad_height = window_size[0] - (matrix.shape[0] % window_size[0])
    pad_width = window_size[1] - (matrix.shape[1] % window_size[1])
    padded = np.pad(matrix, ((0, pad_height), (0, pad_width)), 'constant')
    output = np.zeros((padded.shape[0] // window_size[0], padded.shape[1] // window_size[1]))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = np.max(padded[i*window_size[0]:(i+1)*window_size[0], j*window_size[1]:(j+1)*window_size[1]])
    return output

if __name__ == '__main__':
    img = np.array(Image.open('len_std.jpg').convert('L')) / 255
    kern = np.array([
        [-0.5, 0, 0.5],
        [-1.0, 0, 1.0],
        [-0.5, 0, 0.5],
    ])
    conv1 = conv2d(img, kern)
    relud = np.maximum(conv1, 0)
    pooled = pool(relud, (2, 2))
    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(conv1)
    plt.subplot(223)
    plt.imshow(relud)
    plt.subplot(224)
    plt.imshow(pooled)
    plt.show()

