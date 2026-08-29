import cv2

def clahe_equalization(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
