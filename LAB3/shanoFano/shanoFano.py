import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Change this to "RGB" when you want RGB compression
MODE = "Grayscale"


# ============================================================
# SHANNON-FANO
# ============================================================

def shannon_fano(symbols, codes, prefix=""):
    """
    symbols = [(pixel_value, frequency), ...]
    codes   = {pixel_value: binary_code}
    """

    if len(symbols) == 0:
        return

    if len(symbols) == 1:
        value = symbols[0][0]
        codes[value] = prefix if prefix else "0"
        return

    total = sum(freq for _, freq in symbols)

    current = 0
    split = 0
    min_difference = float("inf")

    for i in range(len(symbols) - 1):
        current += symbols[i][1]

        left_sum = current
        right_sum = total - current

        difference = abs(left_sum - right_sum)

        if difference < min_difference:
            min_difference = difference
            split = i + 1

    left = symbols[:split]
    right = symbols[split:]

    shannon_fano(left, codes, prefix + "0")
    shannon_fano(right, codes, prefix + "1")


# ============================================================
# CREATE SHANNON-FANO CODES
# ============================================================

def create_codes(image):
    pixels = image.flatten()

    values, counts = np.unique(
        pixels,
        return_counts=True
    )

    symbols = list(
        zip(
            values.tolist(),
            counts.tolist()
        )
    )

    # Highest frequency first
    symbols.sort(
        key=lambda x: x[1],
        reverse=True
    )

    codes = {}

    shannon_fano(symbols, codes)

    return codes, symbols


# ============================================================
# ENCODE
# ============================================================

def encode_image(image, codes):
    pixels = image.flatten()

    encoded = []

    for pixel in pixels:
        encoded.append(codes[int(pixel)])

    return "".join(encoded)


# ============================================================
# DECODE
# ============================================================

def decode_image(encoded_bits, codes, shape):
    reverse_codes = {
        code: value
        for value, code in codes.items()
    }

    decoded_pixels = []

    current_code = ""

    for bit in encoded_bits:
        current_code += bit

        if current_code in reverse_codes:
            decoded_pixels.append(
                reverse_codes[current_code]
            )
            current_code = ""

    decoded = np.array(
        decoded_pixels,
        dtype=np.uint8
    )

    return decoded.reshape(shape)


# ============================================================
# PROCESS ONE GRAYSCALE CHANNEL
# ============================================================

def process_channel(channel):
    codes, symbols = create_codes(channel)

    original_bits = channel.size * 8

    encoded_bits = encode_image(
        channel,
        codes
    )

    compressed_bits = len(encoded_bits)

    decoded = decode_image(
        encoded_bits,
        codes,
        channel.shape
    )

    return (
        decoded,
        original_bits,
        compressed_bits,
        len(symbols)
    )


# ============================================================
# MAIN COMPRESSION FUNCTION
# ============================================================

def apply(image, mode):

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if mode == "Grayscale":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        (
            decoded,
            original_bits,
            compressed_bits,
            unique_values
        ) = process_channel(gray)

        compression_ratio = (
            original_bits / compressed_bits
            if compressed_bits > 0
            else 1
        )

        return {
            "image": decoded,
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "unique_values": unique_values
        }

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    else:

        decoded_channels = []

        total_original = 0
        total_compressed = 0
        total_unique = 0

        for channel in range(3):

            current = image[:, :, channel]

            (
                decoded,
                original_bits,
                compressed_bits,
                unique_values
            ) = process_channel(current)

            decoded_channels.append(decoded)

            total_original += original_bits
            total_compressed += compressed_bits
            total_unique += unique_values

        decoded_image = np.stack(
            decoded_channels,
            axis=2
        )

        compression_ratio = (
            total_original / total_compressed
            if total_compressed > 0
            else 1
        )

        return {
            "image": decoded_image,
            "original_bits": total_original,
            "compressed_bits": total_compressed,
            "compression_ratio": compression_ratio,
            "unique_values": total_unique
        }


# ============================================================
# FIND INPUT IMAGE
# ============================================================

script_folder = Path(__file__).resolve().parent

input_path = None

# Look for Input.jpg / Input.jpeg / Input.png / etc.
for file in script_folder.iterdir():

    if not file.is_file():
        continue

    if file.stem.lower() == "input":

        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:
            input_path = file
            break


# ============================================================
# CHECK INPUT
# ============================================================

if input_path is None:

    print("ERROR: Input image not found.")
    print()
    print("Put your image in:")
    print(script_folder)
    print()
    print("The image must be named:")
    print("Input.jpg")
    print("or Input.png")
    print("or Input.jpeg")

    exit()


print("Input file:")
print(input_path)
print()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(input_path),
    cv2.IMREAD_COLOR
)

if image is None:

    print("ERROR: OpenCV could not read the image.")
    print("Check the image file.")

    exit()


# OpenCV gives BGR
# Convert to RGB because our program uses RGB
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


print("Image shape:", image.shape)
print("Image type :", image.dtype)
print()


# ============================================================
# APPLY SHANNON-FANO
# ============================================================

result = apply(
    image,
    MODE
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("======================================")
print("     SHANNON-FANO COMPRESSION")
print("======================================")

print("Mode             :", MODE)
print(
    "Original bits    :",
    result["original_bits"]
)

print(
    "Compressed bits  :",
    result["compressed_bits"]
)

print(
    "Compression ratio:",
    result["compression_ratio"]
)

print(
    "Unique values    :",
    result["unique_values"]
)

print()


# ============================================================
# GET OUTPUT IMAGE
# ============================================================

output = result["image"]


# ============================================================
# SAVE OUTPUT IMAGE
# ============================================================

output_path = script_folder / "Output.png"


if MODE == "RGB":

    # RGB -> BGR for OpenCV
    output_bgr = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

else:

    # Already grayscale
    output_bgr = output


success = cv2.imwrite(
    str(output_path),
    output_bgr
)


if success:

    print("Output image saved at:")
    print(output_path)

else:

    print("ERROR: Could not save output image.")


print()


# ============================================================
# DISPLAY INPUT AND OUTPUT
# ============================================================

if MODE == "RGB":

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
    output_bgr
)


print("Press any key inside the image window to exit.")

cv2.waitKey(0)

cv2.destroyAllWindows()