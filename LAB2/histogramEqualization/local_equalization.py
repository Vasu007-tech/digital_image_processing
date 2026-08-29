import cv2
import numpy as np

def local_equalization(img, tile_size=32):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = gray[i:i+tile_size, j:j+tile_size]
            out[i:i+tile_size, j:j+tile_size] = cv2.equalizeHist(tile)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
