"""
Robust Backlight Panel Rectangle Detection Algorithm

This module detects the largest rectangle (backlight panel) in a grayscale image,
correctly handling rotation. The detection box accurately represents the panel's
actual orientation and boundaries.

Key techniques:
1. Adaptive thresholding for handling uneven illumination
2. Morphological operations for adhesion separation
3. Contour-based detection with cv2.minAreaRect for proper rotation handling
4. Multi-strategy fallback for robustness
"""

import cv2
import numpy as np


def preprocess_image(gray_image, blur_ksize=5):
    """Apply preprocessing to reduce noise while preserving edges.

    Uses bilateral filter for edge-preserving noise reduction, followed by
    a light Gaussian blur to further smooth noise in flat areas.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8).
        blur_ksize: Gaussian blur kernel size (must be odd).

    Returns:
        Preprocessed grayscale image.
    """
    # Bilateral filter: strong noise reduction while preserving edges
    denoised = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
    # Light Gaussian blur for remaining noise
    blurred = cv2.GaussianBlur(denoised, (blur_ksize, blur_ksize), 0)
    return blurred


def adaptive_binarize(gray_image, block_size=51, c_offset=10):
    """Binarize image using Gaussian adaptive threshold.

    Handles uneven illumination by computing threshold locally.

    Args:
        gray_image: Preprocessed grayscale image.
        block_size: Size of local neighborhood for threshold computation (odd, >= 3).
        c_offset: Constant subtracted from the computed threshold.

    Returns:
        Binary image (uint8, values 0 or 255).
    """
    # Ensure block_size is odd and >= 3
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c_offset
    )
    return binary


def separate_adhesion(binary_image, morph_ksize=5, iterations=2):
    """Separate adhesion regions using morphological operations.

    Uses opening (erosion + dilation) to break adhesions, followed by
    closing (dilation + erosion) to fill small gaps.

    Args:
        binary_image: Input binary image.
        morph_ksize: Morphological kernel size.
        iterations: Number of morphological iterations.

    Returns:
        Cleaned binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))

    # Opening: remove small protrusions and break thin connections
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Closing: fill small holes and gaps
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


def _is_valid_rectangle(contour, min_area, max_area, image_shape,
                        min_rectangularity=0.80, max_aspect_ratio=10.0,
                        approx_epsilon=0.02):
    """Validate whether a contour is a valid rectangle candidate.

    Checks area bounds, border touching, rectangularity, aspect ratio,
    and polygon approximation vertex count.

    Args:
        contour: Contour to validate.
        min_area: Minimum allowed area.
        max_area: Maximum allowed area.
        image_shape: Image shape (height, width).
        min_rectangularity: Minimum ratio of contour area to min-area-rect area.
        max_aspect_ratio: Maximum allowed aspect ratio.
        approx_epsilon: Contour approximation epsilon factor.

    Returns:
        True if the contour is a valid rectangle candidate.
    """
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False

    # Check border touching
    if _contour_touches_border(contour, image_shape):
        return False

    # Check rectangularity
    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    if rect_w == 0 or rect_h == 0:
        return False

    rect_area = rect_w * rect_h
    rectangularity = area / rect_area
    if rectangularity < min_rectangularity:
        return False

    # Check aspect ratio
    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
    if aspect_ratio > max_aspect_ratio:
        return False

    # Check polygon approximation (should be close to 4 vertices)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
    if len(approx) < 4 or len(approx) > 8:
        return False

    return True


def find_largest_rectangle(binary_image, min_area_ratio=0.01, max_area_ratio=0.95,
                           approx_epsilon=0.02):
    """Find the largest valid rectangle contour in a binary image.

    Finds contours, validates them as rectangles, and returns the largest one.
    Uses cv2.minAreaRect which correctly computes rotation angle.

    Args:
        binary_image: Input binary image.
        min_area_ratio: Minimum rectangle area as fraction of image area.
        max_area_ratio: Maximum rectangle area as fraction of image area.
        approx_epsilon: Contour approximation epsilon factor.

    Returns:
        A tuple (rect, contour) where rect is ((cx,cy), (w,h), angle) or (None, None).
    """
    h, w = binary_image.shape[:2]
    image_area = h * w
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_rect = None
    best_contour = None
    best_area = 0

    for contour in contours:
        if not _is_valid_rectangle(contour, min_area, max_area,
                                   binary_image.shape,
                                   min_rectangularity=0.75,
                                   approx_epsilon=approx_epsilon):
            continue

        area = cv2.contourArea(contour)
        if area > best_area:
            best_area = area
            best_rect = cv2.minAreaRect(contour)
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

    Uses a multi-strategy approach with fallback mechanisms:
    1. Adaptive threshold + morphology (handles uneven illumination)
    2. Otsu global threshold (first fallback)
    3. Inverted binary (handles bright-target-on-dark-background)
    4. Edge-based detection with Canny (last resort)

    The detected rectangle includes proper rotation information via
    cv2.minAreaRect, which computes the minimum-area rotated bounding box.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8, single channel).
        block_size: Block size for adaptive threshold (must be odd, >= 3).
        c_offset: Constant for adaptive threshold.
        morph_ksize: Morphological kernel size.
        morph_iterations: Number of morphological iterations.
        min_area_ratio: Minimum rectangle area as fraction of image area.
        max_area_ratio: Maximum rectangle area as fraction of image area.
        approx_epsilon: Contour approximation tolerance factor.
        use_otsu_fallback: If True, try Otsu's threshold as fallback.

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

    rect = None
    contour = None
    binary_output = None

    # Strategy 1: Adaptive threshold (handles uneven illumination)
    binary = adaptive_binarize(preprocessed, block_size, c_offset)
    cleaned = separate_adhesion(binary, morph_ksize, morph_iterations)
    binary_output = cleaned

    rect, contour = find_largest_rectangle(
        cleaned, min_area_ratio, max_area_ratio, approx_epsilon
    )

    # Strategy 2: Otsu global threshold
    if rect is None and use_otsu_fallback:
        _, otsu_binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_cleaned = separate_adhesion(otsu_binary, morph_ksize, morph_iterations)

        rect, contour = find_largest_rectangle(
            otsu_cleaned, min_area_ratio, max_area_ratio, approx_epsilon
        )
        if rect is not None:
            binary_output = otsu_cleaned

    # Strategy 3: Inverted binary (target brighter than background)
    if rect is None and use_otsu_fallback:
        _, otsu_binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inverted = cv2.bitwise_not(otsu_binary)
        inv_cleaned = separate_adhesion(inverted, morph_ksize, morph_iterations)

        rect, contour = find_largest_rectangle(
            inv_cleaned, min_area_ratio, max_area_ratio, approx_epsilon
        )
        if rect is not None:
            binary_output = inv_cleaned

    # Strategy 4: Edge-based detection with Canny
    if rect is None:
        edges = cv2.Canny(preprocessed, 50, 150, apertureSize=3)
        # Dilate edges to close gaps
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.dilate(edges, dilate_kernel, iterations=2)
        # Close to form complete contours
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_CLOSE,
                                        close_kernel, iterations=2)

        rect, contour = find_largest_rectangle(
            edges_closed, min_area_ratio, max_area_ratio, approx_epsilon
        )
        if rect is not None:
            binary_output = edges_closed

    # Default binary output if nothing was produced
    if binary_output is None:
        _, binary_output = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
        "binary": binary_output,
    }


def draw_detection_result(color_image, result, color=(0, 255, 0), thickness=2):
    """Draw the detected rectangle on the image.

    Draws the rotated bounding box using the 4 corner points, which correctly
    represents the rotation of the detected rectangle.

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
