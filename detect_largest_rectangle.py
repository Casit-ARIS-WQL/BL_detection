"""
Robust Largest Rectangle Detection Algorithm

This module detects the largest rectangle in a grayscale image where the rectangle
has a significant gray-level difference from the surrounding area. It handles
illumination variations that may cause adhesion between the rectangle and background.

Key techniques:
1. Adaptive thresholding to handle uneven illumination
2. Morphological operations to separate adhesion regions
3. Multi-strategy contour filtering to find the best rectangle candidate
"""

import cv2
import numpy as np


def preprocess_image(gray_image, blur_ksize=5):
    """Apply preprocessing to reduce noise while preserving edges.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8).
        blur_ksize: Kernel size for Gaussian blur (must be odd).

    Returns:
        Preprocessed grayscale image.
    """
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
    # Light Gaussian blur to smooth remaining noise
    blurred = cv2.GaussianBlur(denoised, (blur_ksize, blur_ksize), 0)
    return blurred


def adaptive_binarize(gray_image, block_size=51, c_offset=10):
    """Binarize the image using adaptive thresholding to handle uneven illumination.

    Args:
        gray_image: Preprocessed grayscale image.
        block_size: Size of the neighborhood for adaptive threshold (must be odd).
        c_offset: Constant subtracted from the mean.

    Returns:
        Binary image (0 or 255).
    """
    binary = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_offset,
    )
    return binary


def separate_adhesion(binary_image, morph_ksize=5, iterations=2):
    """Use morphological operations to separate adhesion caused by illumination.

    Args:
        binary_image: Binary image from adaptive thresholding.
        morph_ksize: Kernel size for morphological operations.
        iterations: Number of erosion/dilation iterations.

    Returns:
        Cleaned binary image with adhesion regions separated.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_ksize, morph_ksize)
    )

    # Morphological opening to remove small connections (adhesion)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Morphological closing to fill small gaps within the rectangle
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed


def _contour_touches_border(contour, image_shape, margin=5):
    """Check if a contour touches the image border.

    Args:
        contour: The contour to check.
        image_shape: (height, width) of the image.
        margin: Pixel margin from the border to consider as "touching".

    Returns:
        True if the contour touches any image border.
    """
    h, w = image_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return (x <= margin or y <= margin or
            (x + cw) >= (w - margin) or (y + ch) >= (h - margin))


def find_largest_rectangle(binary_image, min_area_ratio=0.01, max_area_ratio=0.95,
                           approx_epsilon=0.02):
    """Find the largest rectangular contour in the binary image.

    Uses contour approximation and geometric validation to identify the best
    rectangle candidate. Filters out contours that are too large (likely the
    image border) or that touch the image edges.

    Args:
        binary_image: Cleaned binary image.
        min_area_ratio: Minimum contour area as a ratio of image area.
        max_area_ratio: Maximum contour area as a ratio of image area.
            Contours larger than this are rejected as likely being the image border.
        approx_epsilon: Epsilon factor for contour approximation (relative to perimeter).

    Returns:
        A tuple (rect, contour) where rect is ((cx, cy), (w, h), angle) from
        cv2.minAreaRect, and contour is the detected contour. Returns (None, None)
        if no rectangle is found.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    best_rect = None
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # Reject contours that are too large (likely the whole image border)
        if area > max_area:
            continue

        # Reject contours that touch the image border
        if _contour_touches_border(contour, binary_image.shape):
            continue

        # Get minimum area bounding rectangle
        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue
        rect_area = rect_w * rect_h

        # Rectangularity: ratio of contour area to bounding rect area
        rectangularity = area / rect_area

        # Aspect ratio check (reject extremely elongated shapes)
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)

        # Also try polygon approximation
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)

        # A good rectangle candidate should have:
        # - High rectangularity (close to 1.0)
        # - Reasonable aspect ratio
        # - Approximately 4 vertices after polygon approximation
        is_rect_like = (
            rectangularity > 0.75
            and aspect_ratio < 20.0
            and len(approx) >= 4
            and len(approx) <= 8
        )

        if is_rect_like and area > best_area:
            best_area = area
            best_rect = rect
            best_contour = contour

    return best_rect, best_contour


def detect_largest_rectangle(
    gray_image,
    block_size=51,
    c_offset=10,
    morph_ksize=5,
    morph_iterations=2,
    min_area_ratio=0.01,
    max_area_ratio=0.95,
    approx_epsilon=0.02,
    use_otsu_fallback=True,
):
    """Detect the largest rectangle in a grayscale image.

    This is the main entry point. It handles uneven illumination and adhesion
    between the rectangle and surrounding areas. It tries multiple binarization
    strategies (including both normal and inverted thresholds) to handle both
    bright-on-dark and dark-on-bright scenarios.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8, single channel).
        block_size: Block size for adaptive threshold (must be odd, >= 3).
        c_offset: Constant for adaptive threshold.
        morph_ksize: Morphological kernel size for adhesion separation.
        morph_iterations: Number of morphological operation iterations.
        min_area_ratio: Minimum rectangle area as fraction of image area.
        max_area_ratio: Maximum rectangle area as fraction of image area.
            Contours exceeding this are rejected as likely being the image border.
        approx_epsilon: Contour approximation tolerance factor.
        use_otsu_fallback: If True, try Otsu's threshold as fallback when
            adaptive method fails.

    Returns:
        A dictionary with the following keys:
            - "rect": ((center_x, center_y), (width, height), angle) or None
            - "box": numpy array of 4 corner points (int) or None
            - "contour": the detected contour or None
            - "binary": the final binary image used for detection

        Returns None values if no rectangle is detected.

    Raises:
        ValueError: If input image is not a valid grayscale image.
    """
    # Input validation
    if gray_image is None:
        raise ValueError("Input image is None")
    if len(gray_image.shape) != 2:
        raise ValueError("Input image must be a single-channel grayscale image")
    if gray_image.dtype != np.uint8:
        raise ValueError("Input image must be of dtype uint8")

    # Step 1: Preprocess
    preprocessed = preprocess_image(gray_image)

    # Step 2: Try Otsu's threshold (non-inverted) first for bright-on-dark images
    # This directly segments the bright rectangle from the dark background
    rect = None
    contour = None
    cleaned = None

    if use_otsu_fallback:
        _, otsu_binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_cleaned = separate_adhesion(
            otsu_binary, morph_ksize=morph_ksize, iterations=morph_iterations
        )
        rect, contour = find_largest_rectangle(
            otsu_cleaned, min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio, approx_epsilon=approx_epsilon
        )
        if rect is not None:
            cleaned = otsu_cleaned

    # Step 3: Try adaptive binarization (inverted - original behavior)
    if rect is None:
        binary = adaptive_binarize(preprocessed, block_size=block_size, c_offset=c_offset)
        cleaned_adaptive = separate_adhesion(
            binary, morph_ksize=morph_ksize, iterations=morph_iterations
        )
        rect, contour = find_largest_rectangle(
            cleaned_adaptive, min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio, approx_epsilon=approx_epsilon
        )
        if rect is not None:
            cleaned = cleaned_adaptive

    # Step 4: Try inverted adaptive binary (rectangle may be lighter than background)
    if rect is None:
        inverted = cv2.bitwise_not(cleaned_adaptive)
        rect, contour = find_largest_rectangle(
            inverted, min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio, approx_epsilon=approx_epsilon
        )
        if rect is not None:
            cleaned = inverted

    # Step 5: Fallback to Otsu inverted
    if rect is None and use_otsu_fallback:
        _, otsu_binary_inv = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        otsu_inv_cleaned = separate_adhesion(
            otsu_binary_inv, morph_ksize=morph_ksize, iterations=morph_iterations
        )
        rect, contour = find_largest_rectangle(
            otsu_inv_cleaned, min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio, approx_epsilon=approx_epsilon
        )
        if rect is not None:
            cleaned = otsu_inv_cleaned

    # If still nothing found, use the adaptive cleaned as default binary output
    if cleaned is None:
        binary = adaptive_binarize(preprocessed, block_size=block_size, c_offset=c_offset)
        cleaned = separate_adhesion(
            binary, morph_ksize=morph_ksize, iterations=morph_iterations
        )

    # Build result
    box = None
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = np.intp(box)

    return {
        "rect": rect,
        "box": box,
        "contour": contour,
        "binary": cleaned,
    }


def draw_detection_result(color_image, result, color=(0, 255, 0), thickness=2):
    """Draw the detected rectangle on the image.

    Args:
        color_image: BGR color image to draw on (will be modified in place).
        result: Detection result dictionary from detect_largest_rectangle().
        color: BGR color tuple for drawing.
        thickness: Line thickness.

    Returns:
        The image with the rectangle drawn (same reference as input).
    """
    if result["box"] is not None:
        cv2.drawContours(color_image, [result["box"]], 0, color, thickness)
    return color_image


# ----- Example usage -----
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect_largest_rectangle.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "result.png"

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image '{image_path}'")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect
    result = detect_largest_rectangle(gray)

    if result["rect"] is not None:
        center, size, angle = result["rect"]
        print(f"Detected rectangle:")
        print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")
        print(f"  Size: {size[0]:.1f} x {size[1]:.1f}")
        print(f"  Angle: {angle:.1f} degrees")
        print(f"  Box corners: {result['box'].tolist()}")

        # Draw result
        output = img.copy()
        draw_detection_result(output, result)
        cv2.imwrite(output_path, output)
        print(f"Result saved to '{output_path}'")
    else:
        print("No rectangle detected.")
