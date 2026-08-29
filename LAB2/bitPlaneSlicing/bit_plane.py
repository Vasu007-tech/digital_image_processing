import numpy as np
import matplotlib.pyplot as plt
import cv2

def bit_plane_slicing(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, axes = plt.subplots(2,4, figsize=(12,6))
    for i in range(8):
        bit_plane = np.bitwise_and(gray, 1 << i)
        axes[i//4, i%4].imshow(bit_plane, cmap='gray')
        axes[i//4, i%4].set_title(f'Bit Plane {i}')
        axes[i//4, i%4].axis('off')
    return fig
