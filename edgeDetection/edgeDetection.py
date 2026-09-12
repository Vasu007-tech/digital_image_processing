"""
Edge Detection from First Principles (DIP Lab).

This module implements 1st-order and 2nd-order edge detection methods
completely from scratch using pure NumPy.

Input:
    input.jpg / input.png / input.jpeg / input.bmp / input.webp

Output:
    All generated images are saved automatically inside Output/
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# =====================================================================
# 1. First-Principles Greyscale & Spatial Convolution
# =====================================================================

def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image array to uint8 greyscale using luminance weights."""

    if img.ndim == 2:
        return img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] >= 3:
        return (
            0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]
        ).astype(np.uint8)

    return img.astype(np.uint8)


def conv2d_scratch(
    image: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "edge"
) -> np.ndarray:
    """Perform 2D spatial convolution from first principles using NumPy."""

    img_f = image.astype(np.float64)

    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    # Mathematical convolution: flip kernel 180 degrees
    k_rot = np.flipud(np.fliplr(kernel))

    # Pad image
    padded = np.pad(
        img_f,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode=pad_mode
    )

    h, w = img_f.shape

    output = np.zeros(
        (h, w),
        dtype=np.float64
    )

    # Convolution
    for i in range(k_h):
        for j in range(k_w):

            weight = k_rot[i, j]

            if weight != 0.0:
                output += (
                    weight
                    * padded[i:i + h, j:j + w]
                )

    return output


def contrast_stretch_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,255] and convert to uint8."""

    min_v = np.min(arr)
    max_v = np.max(arr)

    if max_v - min_v > 1e-8:
        norm = (
            (arr - min_v)
            / (max_v - min_v)
            * 255.0
        )
    else:
        norm = np.zeros_like(arr)

    return np.clip(
        np.round(norm),
        0,
        255
    ).astype(np.uint8)


# =====================================================================
# 2. First-Order Difference Edge Detection
# =====================================================================

def forward_difference(
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward Difference gradient."""

    img_f = image.astype(np.float64)

    h, w = img_f.shape

    padded = np.pad(
        img_f,
        ((0, 1), (0, 1)),
        mode="edge"
    )

    gx = (
        padded[:h, 1:w + 1]
        - padded[:h, :w]
    )

    gy = (
        padded[1:h + 1, :w]
        - padded[:h, :w]
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    magnitude_uint8 = contrast_stretch_uint8(
        magnitude
    )

    return gx, gy, magnitude_uint8


def backward_difference(
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward Difference gradient."""

    img_f = image.astype(np.float64)

    h, w = img_f.shape

    padded = np.pad(
        img_f,
        ((1, 0), (1, 0)),
        mode="edge"
    )

    gx = (
        padded[1:h + 1, 1:w + 1]
        - padded[1:h + 1, :w]
    )

    gy = (
        padded[1:h + 1, 1:w + 1]
        - padded[:h, 1:w + 1]
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    magnitude_uint8 = contrast_stretch_uint8(
        magnitude
    )

    return gx, gy, magnitude_uint8


def central_difference(
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central Difference gradient."""

    img_f = image.astype(np.float64)

    h, w = img_f.shape

    padded = np.pad(
        img_f,
        ((1, 1), (1, 1)),
        mode="edge"
    )

    gx = (
        padded[1:h + 1, 2:w + 2]
        - padded[1:h + 1, :w]
    ) / 2.0

    gy = (
        padded[2:h + 2, 1:w + 1]
        - padded[:h, 1:w + 1]
    ) / 2.0

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    magnitude_uint8 = contrast_stretch_uint8(
        magnitude
    )

    return gx, gy, magnitude_uint8


# =====================================================================
# 3. Second-Order Edge Detection: LoG and DoG
# =====================================================================

def generate_gaussian_kernel_2d(
    sigma: float,
    size: int | None = None
) -> np.ndarray:

    """Generate a 2D Gaussian smoothing kernel."""

    if size is None:

        radius = int(
            np.ceil(3.0 * sigma)
        )

        size = 2 * radius + 1

    else:

        radius = size // 2

    y, x = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    kernel = np.exp(
        -(x ** 2 + y ** 2)
        / (2.0 * sigma ** 2)
    )

    kernel_sum = np.sum(kernel)

    if kernel_sum > 0:
        kernel /= kernel_sum

    return kernel


def generate_log_kernel(
    sigma: float,
    size: int | None = None
) -> np.ndarray:

    """Generate discrete Laplacian of Gaussian kernel."""

    if size is None:

        radius = int(
            np.ceil(3.0 * sigma)
        )

        size = 2 * radius + 1

    else:

        radius = size // 2

    y, x = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    r2 = (
        x ** 2 + y ** 2
    ).astype(np.float64)

    sigma2 = sigma ** 2
    sigma4 = sigma ** 4

    log_kernel = (
        -(1.0 / (np.pi * sigma4))
        * (
            1.0
            - r2 / (2.0 * sigma2)
        )
        * np.exp(
            -r2 / (2.0 * sigma2)
        )
    )

    # Zero mean
    log_kernel = (
        log_kernel
        - np.mean(log_kernel)
    )

    return log_kernel


def detect_zero_crossings(
    response: np.ndarray,
    threshold_ratio: float = 0.04
) -> np.ndarray:

    """Detect zero crossings of second-order response."""

    max_amp = np.max(
        np.abs(response)
    )

    if max_amp < 1e-8:

        return np.zeros_like(
            response,
            dtype=np.uint8
        )

    tau = threshold_ratio * max_amp

    p = np.pad(
        response,
        1,
        mode="edge"
    )

    c = p[
        1:-1,
        1:-1
    ]

    # Horizontal
    left = p[
        1:-1,
        :-2
    ]

    right = p[
        1:-1,
        2:
    ]

    h_cross = (
        (
            (left * right < 0)
            | (c * left < 0)
            | (c * right < 0)
        )
        & (
            np.abs(left - right)
            > tau
        )
    )

    # Vertical
    up = p[
        :-2,
        1:-1
    ]

    down = p[
        2:,
        1:-1
    ]

    v_cross = (
        (
            (up * down < 0)
            | (c * up < 0)
            | (c * down < 0)
        )
        & (
            np.abs(up - down)
            > tau
        )
    )

    # Main diagonal
    ul = p[
        :-2,
        :-2
    ]

    br = p[
        2:,
        2:
    ]

    d1_cross = (
        (
            (ul * br < 0)
            | (c * ul < 0)
            | (c * br < 0)
        )
        & (
            np.abs(ul - br)
            > tau
        )
    )

    # Anti-diagonal
    ur = p[
        :-2,
        2:
    ]

    bl = p[
        2:,
        :-2
    ]

    d2_cross = (
        (
            (ur * bl < 0)
            | (c * ur < 0)
            | (c * bl < 0)
        )
        & (
            np.abs(ur - bl)
            > tau
        )
    )

    edges = (
        h_cross
        | v_cross
        | d1_cross
        | d2_cross
    )

    return (
        edges.astype(np.uint8)
        * 255
    )


def laplacian_of_gaussian(
    image: np.ndarray,
    sigma: float = 1.4,
    threshold_ratio: float = 0.04
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Execute Laplacian of Gaussian edge detection."""

    kernel = generate_log_kernel(
        sigma
    )

    response = conv2d_scratch(
        image,
        kernel,
        pad_mode="edge"
    )

    response_uint8 = contrast_stretch_uint8(
        response
    )

    zero_crossings = detect_zero_crossings(
        response,
        threshold_ratio
    )

    return (
        response,
        response_uint8,
        zero_crossings
    )


def difference_of_gaussians(
    image: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 1.6,
    threshold_ratio: float = 0.04
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Execute Difference of Gaussians."""

    k1 = generate_gaussian_kernel_2d(
        sigma1
    )

    k2 = generate_gaussian_kernel_2d(
        sigma2
    )

    g1 = conv2d_scratch(
        image,
        k1,
        pad_mode="edge"
    )

    g2 = conv2d_scratch(
        image,
        k2,
        pad_mode="edge"
    )

    dog_response = g1 - g2

    response_uint8 = contrast_stretch_uint8(
        dog_response
    )

    zero_crossings = detect_zero_crossings(
        dog_response,
        threshold_ratio
    )

    return (
        dog_response,
        response_uint8,
        zero_crossings
    )


# =====================================================================
# 4. Grid Assembly Utilities
# =====================================================================

def to_3ch(
    img_2d: np.ndarray
) -> np.ndarray:

    """Convert 2D image to 3-channel image."""

    if img_2d.ndim == 3:
        return img_2d

    return np.stack(
        [
            img_2d,
            img_2d,
            img_2d
        ],
        axis=-1
    )


def build_mosaic_grid(
    images: list[np.ndarray],
    rows: int,
    cols: int,
    border_width: int = 2,
    border_color: int = 100
) -> np.ndarray:

    """Create image comparison grid."""

    if not images:
        raise ValueError(
            "Image list cannot be empty."
        )

    h, w = images[0].shape[:2]

    std_images = []

    for im in images:

        if im.shape[:2] != (h, w):

            pil_im = Image.fromarray(im)

            pil_im = pil_im.resize(
                (w, h),
                Image.Resampling.BILINEAR
            )

            im = np.array(pil_im)

        std_images.append(
            to_3ch(im)
        )

    total_slots = rows * cols

    while len(std_images) < total_slots:

        std_images.append(
            np.zeros(
                (h, w, 3),
                dtype=np.uint8
            )
        )

    row_blocks = []

    idx = 0

    for _ in range(rows):

        col_cells = []

        for c in range(cols):

            col_cells.append(
                std_images[idx]
            )

            if c < cols - 1:

                col_cells.append(
                    np.full(
                        (
                            h,
                            border_width,
                            3
                        ),
                        border_color,
                        dtype=np.uint8
                    )
                )

            idx += 1

        row_assembled = np.hstack(
            col_cells
        )

        row_blocks.append(
            row_assembled
        )

    grid_rows = []

    for r in range(rows):

        grid_rows.append(
            row_blocks[r]
        )

        if r < rows - 1:

            sep_h = np.full(
                (
                    border_width,
                    row_blocks[r].shape[1],
                    3
                ),
                border_color,
                dtype=np.uint8
            )

            grid_rows.append(
                sep_h
            )

    return np.vstack(
        grid_rows
    )


# =====================================================================
# 5. Main Execution
# =====================================================================

def run_edge_detection(
    input_image_path: str | None = None,
    output_directory: str = "Output"
) -> dict[str, str]:

    """Execute complete edge detection workflow."""

    # Folder containing this Python file
    script_dir = Path(
        __file__
    ).resolve().parent

    # ---------------------------------------------------------------
    # Find input image
    # ---------------------------------------------------------------

    if input_image_path is None:

        valid_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp"
        )

        candidates = [
            f
            for f in script_dir.iterdir()
            if (
                f.is_file()
                and f.suffix.lower()
                in valid_exts
                and f.stem.lower() == "input"
            )
        ]

        if not candidates:

            raise FileNotFoundError(
                "Input image not found.\n"
                "Put input.jpg / input.png / input.jpeg "
                "in:\n"
                f"{script_dir}"
            )

        input_image_path = str(
            candidates[0]
        )

    print(
        "=" * 70
    )

    print(
        "DIP LAB - EDGE DETECTION PIPELINE"
    )

    print(
        "FROM FIRST PRINCIPLES"
    )

    print(
        "=" * 70
    )

    print(
        f"Loading input image: {input_image_path}"
    )

    # ---------------------------------------------------------------
    # Read image
    # ---------------------------------------------------------------

    pil_orig = Image.open(
        input_image_path
    ).convert("RGB")

    rgb_arr = np.array(
        pil_orig
    )

    grey_arr = rgb_to_grey(
        rgb_arr
    )

    print(
        f"Image Dimensions: "
        f"{grey_arr.shape[1]}x"
        f"{grey_arr.shape[0]} px"
    )

    # ---------------------------------------------------------------
    # Create Output folder
    # ---------------------------------------------------------------

    out_dir = (
        script_dir
        / output_directory
    )

    out_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    saved_files = {}

    # ---------------------------------------------------------------
    # Save grayscale input
    # ---------------------------------------------------------------

    p_grey = (
        out_dir
        / "edge_input_greyscale.png"
    )

    Image.fromarray(
        to_3ch(grey_arr)
    ).save(p_grey)

    saved_files[
        "Input Greyscale"
    ] = str(p_grey)

    # ---------------------------------------------------------------
    # First-Order Operators
    # ---------------------------------------------------------------

    print()
    print(
        "[1/3] Computing First-Order "
        "Numerical Differences..."
    )

    # Forward
    t0 = time.perf_counter()

    _, _, fwd_mag = forward_difference(
        grey_arr
    )

    t_fwd = (
        time.perf_counter()
        - t0
    )

    print(
        f"  [OK] Forward Difference: "
        f"{t_fwd * 1000:.2f} ms"
    )

    # Backward
    t0 = time.perf_counter()

    _, _, bwd_mag = backward_difference(
        grey_arr
    )

    t_bwd = (
        time.perf_counter()
        - t0
    )

    print(
        f"  [OK] Backward Difference: "
        f"{t_bwd * 1000:.2f} ms"
    )

    # Central
    t0 = time.perf_counter()

    _, _, cen_mag = central_difference(
        grey_arr
    )

    t_cen = (
        time.perf_counter()
        - t0
    )

    print(
        f"  [OK] Central Difference: "
        f"{t_cen * 1000:.2f} ms"
    )

    # Save
    p_fwd = (
        out_dir
        / "edge_1st_order_forward.png"
    )

    p_bwd = (
        out_dir
        / "edge_1st_order_backward.png"
    )

    p_cen = (
        out_dir
        / "edge_1st_order_central.png"
    )

    Image.fromarray(
        to_3ch(fwd_mag)
    ).save(p_fwd)

    Image.fromarray(
        to_3ch(bwd_mag)
    ).save(p_bwd)

    Image.fromarray(
        to_3ch(cen_mag)
    ).save(p_cen)

    saved_files[
        "1st Order Forward"
    ] = str(p_fwd)

    saved_files[
        "1st Order Backward"
    ] = str(p_bwd)

    saved_files[
        "1st Order Central"
    ] = str(p_cen)

    # ---------------------------------------------------------------
    # LoG and DoG
    # ---------------------------------------------------------------

    print()
    print(
        "[2/3] Computing Second-Order "
        "Operators & Zero-Crossings..."
    )

    # LoG
    t0 = time.perf_counter()

    (
        _,
        log_resp_u8,
        log_zc
    ) = laplacian_of_gaussian(
        grey_arr,
        sigma=1.4,
        threshold_ratio=0.04
    )

    t_log = (
        time.perf_counter()
        - t0
    )

    print(
        f"  [OK] LoG: "
        f"{t_log * 1000:.2f} ms"
    )

    # DoG
    t0 = time.perf_counter()

    (
        _,
        dog_resp_u8,
        dog_zc
    ) = difference_of_gaussians(
        grey_arr,
        sigma1=1.0,
        sigma2=1.6,
        threshold_ratio=0.04
    )

    t_dog = (
        time.perf_counter()
        - t0
    )

    print(
        f"  [OK] DoG: "
        f"{t_dog * 1000:.2f} ms"
    )

    # Save second-order outputs
    p_log_resp = (
        out_dir
        / "edge_2nd_order_log_response.png"
    )

    p_log_zc = (
        out_dir
        / "edge_2nd_order_log_zerocrossing.png"
    )

    p_dog_resp = (
        out_dir
        / "edge_2nd_order_dog_response.png"
    )

    p_dog_zc = (
        out_dir
        / "edge_2nd_order_dog_zerocrossing.png"
    )

    Image.fromarray(
        to_3ch(log_resp_u8)
    ).save(p_log_resp)

    Image.fromarray(
        to_3ch(log_zc)
    ).save(p_log_zc)

    Image.fromarray(
        to_3ch(dog_resp_u8)
    ).save(p_dog_resp)

    Image.fromarray(
        to_3ch(dog_zc)
    ).save(p_dog_zc)

    saved_files[
        "LoG Response"
    ] = str(p_log_resp)

    saved_files[
        "LoG Zero-Crossings"
    ] = str(p_log_zc)

    saved_files[
        "DoG Response"
    ] = str(p_dog_resp)

    saved_files[
        "DoG Zero-Crossings"
    ] = str(p_dog_zc)

    # ---------------------------------------------------------------
    # Comparison grids
    # ---------------------------------------------------------------

    print()
    print(
        "[3/3] Generating Comparison Grids..."
    )

    # 1st-order grid
    grid_1st = build_mosaic_grid(
        [
            grey_arr,
            fwd_mag,
            bwd_mag,
            cen_mag
        ],
        rows=1,
        cols=4
    )

    p_grid_1st = (
        out_dir
        / "edge_comparison_1st_order.png"
    )

    Image.fromarray(
        grid_1st
    ).save(p_grid_1st)

    saved_files[
        "1st Order Comparison Grid"
    ] = str(p_grid_1st)

    # 2nd-order grid
    grid_2nd = build_mosaic_grid(
        [
            log_resp_u8,
            log_zc,
            dog_resp_u8,
            dog_zc
        ],
        rows=2,
        cols=2
    )

    p_grid_2nd = (
        out_dir
        / "edge_comparison_2nd_order.png"
    )

    Image.fromarray(
        grid_2nd
    ).save(p_grid_2nd)

    saved_files[
        "2nd Order Comparison Grid"
    ] = str(p_grid_2nd)

    # Master grid
    grid_master = build_mosaic_grid(
        [
            grey_arr,
            fwd_mag,
            bwd_mag,
            cen_mag,
            log_resp_u8,
            log_zc,
            dog_resp_u8,
            dog_zc
        ],
        rows=2,
        cols=4
    )

    p_grid_master = (
        out_dir
        / "edge_detection_master_grid.png"
    )

    Image.fromarray(
        grid_master
    ).save(p_grid_master)

    saved_files[
        "Master Overview Grid"
    ] = str(p_grid_master)

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EDGE DETECTION COMPLETED SUCCESSFULLY!"
    )

    print(
        "=" * 70
    )

    print(
        "Generated files:"
    )

    for title, path in saved_files.items():

        print(
            f"  * {title:30s} -> {path}"
        )

    print(
        "=" * 70
    )

    return saved_files


# =====================================================================
# PROGRAM START
# =====================================================================

if __name__ == "__main__":

    target_img = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    run_edge_detection(
        target_img
    )