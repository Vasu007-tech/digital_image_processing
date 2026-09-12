import os
from pathlib import Path

import numpy as np
import cv2
import heapq
from collections import Counter


NAME = "Huffman Compression"

PARAMS = {
    "channel_mode": {
        "type": "select",
        "label": "Image Mode",
        "options": [
            "Grayscale",
            "RGB"
        ],
        "default": "Grayscale"
    }
}


# ============================================================
# HUFFMAN NODE
# ============================================================

class Node:

    def __init__(
        self,
        symbol=None,
        frequency=0,
        left=None,
        right=None
    ):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency


# ============================================================
# BUILD HUFFMAN TREE
# ============================================================

def build_tree(image):

    pixels = image.flatten()

    frequency = Counter(
        pixels.tolist()
    )

    heap = []

    for symbol, freq in frequency.items():

        node = Node(
            symbol=symbol,
            frequency=freq
        )

        heapq.heappush(
            heap,
            node
        )

    if len(heap) == 1:
        return heap[0]

    while len(heap) > 1:

        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        parent = Node(
            frequency=(
                left.frequency
                + right.frequency
            ),
            left=left,
            right=right
        )

        heapq.heappush(
            heap,
            parent
        )

    return heap[0]


# ============================================================
# GENERATE HUFFMAN CODES
# ============================================================

def generate_codes(
    node,
    codes,
    prefix=""
):

    if node is None:
        return

    if (
        node.left is None
        and node.right is None
    ):

        codes[node.symbol] = (
            prefix
            if prefix
            else "0"
        )

        return

    generate_codes(
        node.left,
        codes,
        prefix + "0"
    )

    generate_codes(
        node.right,
        codes,
        prefix + "1"
    )


# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image(image):

    tree = build_tree(image)

    codes = {}

    generate_codes(
        tree,
        codes
    )

    pixels = image.flatten()

    encoded_bits = sum(
        len(codes[int(value)])
        for value in pixels
    )

    original_bits = (
        image.size * 8
    )

    return (
        codes,
        original_bits,
        encoded_bits
    )


# ============================================================
# APPLY HUFFMAN COMPRESSION
# ============================================================

def apply(image, params):

    mode = params["channel_mode"]

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if mode == "Grayscale":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        (
            codes,
            original_bits,
            encoded_bits
        ) = encode_image(gray)

        compression_ratio = (
            original_bits / encoded_bits
            if encoded_bits > 0
            else 1
        )

        return {
            "image": gray,
            "original_bits": original_bits,
            "compressed_bits": encoded_bits,
            "compression_ratio": compression_ratio,
            "unique_values": len(codes)
        }

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    else:

        total_original = 0
        total_encoded = 0
        total_unique = 0

        for channel in range(3):

            current = image[:, :, channel]

            (
                codes,
                original_bits,
                encoded_bits
            ) = encode_image(
                current
            )

            total_original += original_bits
            total_encoded += encoded_bits
            total_unique += len(codes)

        compression_ratio = (
            total_original / total_encoded
            if total_encoded > 0
            else 1
        )

        return {
            "image": image,
            "original_bits": total_original,
            "compressed_bits": total_encoded,
            "compression_ratio": compression_ratio,
            "unique_values": total_unique
        }


# ============================================================
# FIND INPUT IMAGE
# ============================================================

script_dir = Path(
    __file__
).resolve().parent

input_path = None

valid_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

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
    print("input.jpg")
    print("or input.png")
    print("or input.jpeg")

    exit()


print(
    "Input image:",
    input_path
)


# ============================================================
# READ IMAGE
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


# OpenCV loads BGR
# Convert to RGB
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


print(
    "Image shape:",
    image.shape
)

print(
    "Image type:",
    image.dtype
)


# ============================================================
# SELECT MODE
# ============================================================

params = {
    "channel_mode": "Grayscale"
}

# For RGB compression, use:
#
# params = {
#     "channel_mode": "RGB"
# }


# ============================================================
# RUN HUFFMAN COMPRESSION
# ============================================================

result = apply(
    image,
    params
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 50)
print("       HUFFMAN COMPRESSION")
print("=" * 50)

print(
    "Mode              :",
    params["channel_mode"]
)

print(
    "Original bits     :",
    result["original_bits"]
)

print(
    "Compressed bits   :",
    result["compressed_bits"]
)

print(
    "Compression ratio :",
    result["compression_ratio"]
)

print(
    "Unique values     :",
    result["unique_values"]
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output = result["image"]

output_path = (
    script_dir
    / "Output.png"
)


if output.ndim == 3:

    output_to_save = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

else:

    output_to_save = output


success = cv2.imwrite(
    str(output_path),
    output_to_save
)


if success:

    print()
    print(
        "Output saved at:",
        output_path
    )

else:

    print()
    print(
        "ERROR: Could not save output."
    )


# ============================================================
# DISPLAY IMAGE
# ============================================================

if params["channel_mode"] == "RGB":

    input_display = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

else:

    input_display = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )


cv2.namedWindow(
    "Input Image",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Output Image",
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    "Input Image",
    800,
    600
)

cv2.resizeWindow(
    "Output Image",
    800,
    600
)


cv2.imshow(
    "Input Image",
    input_display
)

cv2.imshow(
    "Output Image",
    output_to_save
)


print()
print(
    "Press any key in the image window to exit."
)

cv2.waitKey(0)

cv2.destroyAllWindows()