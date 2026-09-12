import os
from pathlib import Path

import cv2
import numpy as np
import pywt


NAME = "Haar Wavelet Transform"

PARAMS = {
    "level": {
        "type": "slider",
        "label": "Decomposition Level",
        "min": 1,
        "max": 3,
        "step": 1,
        "default": 1
    },

    "component": {
        "type": "select",
        "label": "Component",
        "options": [
            "Approximation",
            "Horizontal Detail",
            "Vertical Detail",
            "Diagonal Detail"
        ],
        "default": "Approximation"
    }
}


# ============================================================
# HAAR WAVELET TRANSFORM
# ============================================================

def apply(image, params):

    level = params["level"]
    component = params["component"]

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Haar wavelet decomposition
    coefficients = pywt.wavedec2(
        gray.astype(np.float64),
        "haar",
        level=level
    )

    # --------------------------------------------------------
    # APPROXIMATION
    # --------------------------------------------------------

    approximation = coefficients[0]

    if component == "Approximation":

        result = approximation

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    else:

        # coefficients structure:
        #
        # coefficients[0] = approximation at highest level
        # coefficients[1] = details at highest level
        # coefficients[2] = details at next level
        # ...
        #
        # Therefore:
        # level 1 -> coefficients[-1]
        # level 2 -> coefficients[-2]
        # level 3 -> coefficients[-3]

        horizontal, vertical, diagonal = (
            coefficients[-1]
        )

        if component == "Horizontal Detail":

            result = horizontal

        elif component == "Vertical Detail":

            result = vertical

        else:

            result = diagonal

    # --------------------------------------------------------
    # NORMALIZE RESULT FOR DISPLAY
    # --------------------------------------------------------

    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    result = result.astype(
        np.uint8
    )

    return result


# ============================================================
# FIND INPUT IMAGE
# ============================================================

script_dir = Path(
    __file__
).resolve().parent

input_path = None

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

for file in script_dir.iterdir():

    if (
        file.is_file()
        and file.stem.lower() == "input"
        and file.suffix.lower()
        in valid_extensions
    ):
        input_path = file
        break


# ============================================================
# CHECK INPUT
# ============================================================

if input_path is None:

    print("ERROR: Input image not found.")
    print()
    print("Put your image in:")
    print(script_dir)
    print()
    print("The image should be named:")
    print("Input.jpg")
    print("or Input.png")
    print("or Input.jpeg")

    exit()


print("=" * 60)
print("       HAAR WAVELET TRANSFORM")
print("=" * 60)

print(
    "Input image:",
    input_path
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(input_path),
    cv2.IMREAD_COLOR
)

if image is None:

    print(
        "ERROR: Could not read input image."
    )

    exit()


# OpenCV reads BGR
# Convert to RGB
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


print(
    "Image shape:",
    image.shape
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_dir = (
    script_dir
    / "Output"
)

output_dir.mkdir(
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

level = 1

# Change this to:
#
# 1
# 2
# 3
#
# depending on the decomposition level.

components = [
    "Approximation",
    "Horizontal Detail",
    "Vertical Detail",
    "Diagonal Detail"
]


# ============================================================
# GENERATE RESULTS
# ============================================================

results = {}


for component in components:

    params = {
        "level": level,
        "component": component
    }

    result = apply(
        image,
        params
    )

    results[component] = result


# ============================================================
# SAVE OUTPUTS
# ============================================================

for component, result in results.items():

    filename = (
        component
        .lower()
        .replace(" ", "_")
        + f"_level_{level}.png"
    )

    output_path = (
        output_dir
        / filename
    )

    cv2.imwrite(
        str(output_path),
        result
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# CREATE DISPLAY GRID
# ============================================================

approximation = results[
    "Approximation"
]

horizontal = results[
    "Horizontal Detail"
]

vertical = results[
    "Vertical Detail"
]

diagonal = results[
    "Diagonal Detail"
]


# Put all four images into one grid

top_row = np.hstack([
    approximation,
    horizontal
])

bottom_row = np.hstack([
    vertical,
    diagonal
])

grid = np.vstack([
    top_row,
    bottom_row
])


# Save grid
grid_path = (
    output_dir
    / f"wavelet_comparison_level_{level}.png"
)

cv2.imwrite(
    str(grid_path),
    grid
)

print(
    f"Saved: {grid_path}"
)


# ============================================================
# DISPLAY
# ============================================================

cv2.namedWindow(
    "Haar Wavelet Transform",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Haar Wavelet Transform",
    1000,
    700
)

cv2.imshow(
    "Haar Wavelet Transform",
    grid
)

print()
print("Press any key to close the window.")

cv2.waitKey(0)

cv2.destroyAllWindows()


print()
print("=" * 60)
print("WAVELET TRANSFORM COMPLETED")
print("=" * 60)