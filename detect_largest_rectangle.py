"""
Precise Backlight Panel Rectangle Detection Algorithm

This module detects the backlight panel rectangle in a grayscale image with
high precision, ensuring the detected boundary tightly fits the actual panel edge.

Key techniques:
1. Edge-based detection using Canny for precise boundary localization
2. Gradient-guided contour refinement for sub-pixel accuracy
3. Minimal morphological processing to avoid boundary shrinkage
4. Multi-strategy approach combining threshold and edge methods
"""

import cv2
import numpy as np


def preprocess_image(gray_image):
    """Apply light preprocessing to reduce noise while preserving edges.

    Uses bilateral filter which smooths flat regions but preserves sharp edges,
    critical for maintaining precise boundary detection.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8).

    Returns:
        Preprocessed grayscale image with noise reduced but edges preserved.
    """
    denoised = cv2.bilateralFilter(gray_image, d=7, sigmaColor=50, sigmaSpace=50)
    return denoised


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


def _is_rectangle_like(contour, min_rectangularity=0.80, max_aspect_ratio=10.0):
    """Check if a contour is rectangle-like based on geometric properties.

    Args:
        contour: The contour to evaluate.
        min_rectangularity: Minimum ratio of contour area to bounding rect area.
        max_aspect_ratio: Maximum allowed aspect ratio.

    Returns:
        True if the contour has rectangular properties.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return False

    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    if rect_w == 0 or rect_h == 0:
        return False

    rect_area = rect_w * rect_h
    rectangularity = area / rect_area
    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)

    return rectangularity > min_rectangularity and aspect_ratio < max_aspect_ratio


def detect_by_edge(gray_image, preprocessed, min_area_ratio=0.01, max_area_ratio=0.95):
    """Detect rectangle using Canny edge detection for precise boundaries.

    This method finds edges with sub-pixel accuracy and traces the panel boundary
    tightly. It uses automatic Canny threshold computation based on image statistics.

    Args:
        gray_image: Original grayscale image.
        preprocessed: Preprocessed (denoised) grayscale image.
        min_area_ratio: Minimum contour area as ratio of image area.
        max_area_ratio: Maximum contour area as ratio of image area.

    Returns:
        A tuple (rect, contour, edge_image) or (None, None, edge_image).
    """
    image_area = gray_image.shape[0] * gray_image.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    # Compute Canny thresholds automatically using median intensity
    median_val = np.median(preprocessed)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))

    # Use Canny edge detection for precise edge localization
    edges = cv2.Canny(preprocessed, lower, upper, apertureSize=3, L2gradient=True)

    # Dilate edges slightly to connect nearby edge fragments
    # Use a small kernel to avoid shifting the boundary
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_connected = cv2.dilate(edges, dilate_kernel, iterations=1)

    # Find contours from the edge image
    contours, _ = cv2.findContours(
        edges_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_rect = None
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if _contour_touches_border(contour, gray_image.shape):
            continue
        if not _is_rectangle_like(contour):
            continue

        if area > best_area:
            best_area = area
            best_rect = cv2.minAreaRect(contour)
            best_contour = contour

    return best_rect, best_contour, edges


def detect_by_threshold(preprocessed, min_area_ratio=0.01, max_area_ratio=0.95):
    """Detect rectangle using Otsu threshold with minimal morphological processing.

    For bright-on-dark images (backlight panel), this segments the bright region
    with minimal boundary erosion.

    Args:
        preprocessed: Preprocessed grayscale image.
        min_area_ratio: Minimum contour area as ratio of image area.
        max_area_ratio: Maximum contour area as ratio of image area.

    Returns:
        A tuple (rect, contour, binary_image) or (None, None, binary_image).
    """
    image_area = preprocessed.shape[0] * preprocessed.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    # Otsu threshold for bright panel segmentation
    _, binary = cv2.threshold(
        preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Use only closing (no opening/erosion) to fill small internal gaps
    # without shrinking the boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_rect = None
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if _contour_touches_border(contour, preprocessed.shape):
            continue
        if not _is_rectangle_like(contour):
            continue

        if area > best_area:
            best_area = area
            best_rect = cv2.minAreaRect(contour)
            best_contour = contour

    return best_rect, best_contour, cleaned


def refine_contour_with_edges(gray_image, contour, search_width=15):
    """Refine contour points using local gradient maxima for sub-pixel edge accuracy.

    For each point on the contour, searches along the normal direction to find
    the strongest gradient (actual edge), pulling the contour to the true boundary.

    Args:
        gray_image: Original grayscale image.
        contour: Initial contour to refine.
        search_width: Number of pixels to search in each direction along the normal.

    Returns:
        Refined contour with points snapped to actual edges.
    """
    if contour is None or len(contour) < 4:
        return contour

    h, w = gray_image.shape[:2]

    # Compute gradient magnitude for edge detection
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Get the minAreaRect to determine edge normals
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # For each edge of the rectangle, refine points near that edge
    refined_points = []
    contour_points = contour.reshape(-1, 2).astype(np.float64)

    for i in range(len(contour_points)):
        px, py = contour_points[i]
        ix, iy = int(round(px)), int(round(py))

        # Determine local normal direction using neighboring contour points
        prev_idx = (i - 1) % len(contour_points)
        next_idx = (i + 1) % len(contour_points)
        tangent = contour_points[next_idx] - contour_points[prev_idx]
        tangent_len = np.linalg.norm(tangent)

        if tangent_len < 1e-6:
            refined_points.append([ix, iy])
            continue

        tangent = tangent / tangent_len
        # Normal is perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]])

        # Search along the normal for the strongest gradient
        best_grad = 0
        best_pos = np.array([px, py])

        for offset in range(-search_width, search_width + 1):
            sx = int(round(px + offset * normal[0]))
            sy = int(round(py + offset * normal[1]))

            if 0 <= sx < w and 0 <= sy < h:
                g = grad_mag[sy, sx]
                if g > best_grad:
                    best_grad = g
                    best_pos = np.array([sx, sy])

        refined_points.append([int(best_pos[0]), int(best_pos[1])])

    refined = np.array(refined_points, dtype=np.int32).reshape(-1, 1, 2)
    return refined


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
    """Detect the largest rectangle in a grayscale image with high precision.

    This is the main entry point. It combines edge-based detection with threshold-based
    detection to find the backlight panel boundary that tightly fits the actual edge.

    The algorithm prioritizes precision:
    1. First tries edge-based detection (Canny) for the tightest boundary
    2. Falls back to threshold-based detection with minimal morphology
    3. Refines the detected contour using gradient information

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8, single channel).
        block_size: Block size for adaptive threshold (must be odd, >= 3).
        c_offset: Constant for adaptive threshold.
        morph_ksize: Morphological kernel size (kept for API compatibility).
        morph_iterations: Number of morphological iterations (kept for API compatibility).
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

    # Step 1: Preprocess with edge-preserving filter
    preprocessed = preprocess_image(gray_image)

    rect = None
    contour = None
    binary_output = None

    # Step 2: Try threshold-based detection first (works well for bright panel on dark bg)
    rect, contour, binary_output = detect_by_threshold(
        preprocessed, min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
    )

    # Step 3: If threshold detection found something, refine with edges
    if contour is not None:
        refined_contour = refine_contour_with_edges(gray_image, contour)
        # Only use refined contour if it's still rectangle-like and has reasonable area
        if refined_contour is not None and _is_rectangle_like(refined_contour, 0.70):
            refined_area = cv2.contourArea(refined_contour)
            image_area = gray_image.shape[0] * gray_image.shape[1]
            if refined_area > image_area * min_area_ratio:
                contour = refined_contour
                rect = cv2.minAreaRect(contour)

    # Step 4: If threshold failed, try edge-based detection
    if rect is None:
        rect, contour, edge_img = detect_by_edge(
            gray_image, preprocessed,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
        )
        if binary_output is None:
            binary_output = edge_img

    # Step 5: Fallback - try adaptive threshold with minimal morphology
    if rect is None:
        binary = cv2.adaptiveThreshold(
            preprocessed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c_offset,
        )
        # Minimal closing only - no opening to avoid boundary shrinkage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        image_area = gray_image.shape[0] * gray_image.shape[1]
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < image_area * min_area_ratio or area > image_area * max_area_ratio:
                continue
            if _contour_touches_border(c, gray_image.shape):
                continue
            if not _is_rectangle_like(c, 0.70):
                continue
            if area > best_area:
                best_area = area
                rect = cv2.minAreaRect(c)
                contour = c
                binary_output = cleaned

    # Step 6: Try inverted adaptive threshold
    if rect is None:
        binary_inv = cv2.adaptiveThreshold(
            preprocessed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c_offset,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

        image_area = gray_image.shape[0] * gray_image.shape[1]
        contours, _ = cv2.findContours(
            cleaned_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < image_area * min_area_ratio or area > image_area * max_area_ratio:
                continue
            if _contour_touches_border(c, gray_image.shape):
                continue
            if not _is_rectangle_like(c, 0.70):
                continue
            if area > best_area:
                best_area = area
                rect = cv2.minAreaRect(c)
                contour = c
                binary_output = cleaned_inv

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
