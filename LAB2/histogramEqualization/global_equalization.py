import cv2

def global_equalization(img):
    if img is None:
        return None
    # Convert to grayscale for equalization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    # Convert back to RGB for Streamlit
    return cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
