"""
Precise Backlight Panel Rectangle Detection Algorithm

This module detects the backlight panel rectangle in a grayscale image with
high precision, ensuring the detected boundary tightly fits the actual inner
panel edge (the bright illuminated area), not the outer frame.

Key techniques:
1. High-percentile thresholding to isolate only the brightest region (panel)
2. Erosion-based boundary tightening to exclude semi-bright frame pixels
3. Edge-based refinement that pulls boundaries INWARD using gradient direction
4. Multi-strategy approach with strict bright-region isolation
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
    denoised = cv2.bilateralFilter(gray_image, d=5, sigmaColor=40, sigmaSpace=40)
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


def _compute_bright_threshold(gray_image):
    """Compute a threshold that isolates the bright panel from the frame.

    Uses the intensity distribution to find a threshold above which only
    the actual bright panel pixels exist, excluding the semi-bright frame.

    Args:
        gray_image: Grayscale image.

    Returns:
        Threshold value (int).
    """
    # Use a high percentile to find the intensity of the bright panel
    # The panel is very bright (near 255), while the frame is dimmer
    high_val = np.percentile(gray_image, 85)
    low_val = np.percentile(gray_image, 30)

    # The threshold should be set between frame brightness and panel brightness
    # Use a weighted value closer to the bright region
    threshold = int(low_val + 0.7 * (high_val - low_val))

    # Ensure minimum threshold for very bright panels
    threshold = max(threshold, 180)

    return threshold


def detect_by_bright_threshold(gray_image, preprocessed,
                               min_area_ratio=0.01, max_area_ratio=0.95):
    """Detect rectangle by isolating only the brightest region (inner panel).

    Uses a high threshold to select only the actual illuminated panel area,
    then applies light erosion to tighten the boundary to the true inner edge.

    Args:
        gray_image: Original grayscale image.
        preprocessed: Preprocessed (denoised) grayscale image.
        min_area_ratio: Minimum contour area as ratio of image area.
        max_area_ratio: Maximum contour area as ratio of image area.

    Returns:
        A tuple (rect, contour, binary_image) or (None, None, binary_image).
    """
    image_area = preprocessed.shape[0] * preprocessed.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    # Compute a high threshold that isolates only the bright panel
    thresh_val = _compute_bright_threshold(preprocessed)
    _, binary = cv2.threshold(preprocessed, thresh_val, 255, cv2.THRESH_BINARY)

    # Apply small erosion to tighten boundary away from semi-bright frame pixels
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.erode(binary, erode_kernel, iterations=1)

    # Close small gaps within the panel
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

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
        if not _is_rectangle_like(contour, min_rectangularity=0.75):
            continue

        if area > best_area:
            best_area = area
            best_rect = cv2.minAreaRect(contour)
            best_contour = contour

    return best_rect, best_contour, cleaned


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

    # Use higher Canny thresholds to detect only strong edges (panel boundary)
    median_val = np.median(preprocessed)
    lower = int(max(30, 0.6 * median_val))
    upper = int(min(255, 1.8 * median_val))

    edges = cv2.Canny(preprocessed, lower, upper, apertureSize=3, L2gradient=True)

    # Use minimal dilation to connect edge fragments
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_connected = cv2.dilate(edges, dilate_kernel, iterations=1)

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


def refine_rect_with_edge_scan(gray_image, rect, scan_range=20):
    """Refine rectangle edges by scanning for the steepest intensity drop.

    For each side of the rectangle, scans inward from the current boundary
    to find where the intensity drops sharply (transition from bright panel
    to darker frame), ensuring the boundary is at the INNER edge.

    Args:
        gray_image: Original grayscale image.
        rect: The initial minAreaRect ((cx,cy), (w,h), angle).
        scan_range: Number of pixels to scan inward from each edge.

    Returns:
        Refined minAreaRect or the original if refinement fails.
    """
    if rect is None:
        return rect

    h, w = gray_image.shape[:2]
    center, size, angle = rect
    cx, cy = center
    rw, rh = size

    # Get the box points to determine edge directions
    box = cv2.boxPoints(rect)

    # Compute inward normals for each edge
    # Each edge is defined by two adjacent box points
    refined_offsets = [0.0, 0.0, 0.0, 0.0]  # shrink for each side

    for edge_idx in range(4):
        p1 = box[edge_idx]
        p2 = box[(edge_idx + 1) % 4]

        # Edge midpoint
        mid = (p1 + p2) / 2.0

        # Edge direction and inward normal
        edge_dir = p2 - p1
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1:
            continue

        edge_dir = edge_dir / edge_len
        # Inward normal (pointing toward rectangle center)
        normal = np.array([-edge_dir[1], edge_dir[0]])

        # Ensure normal points inward (toward center)
        to_center = np.array([cx - mid[0], cy - mid[1]])
        if np.dot(normal, to_center) < 0:
            normal = -normal

        # Sample intensity profile along inward normal from the edge midpoint
        intensities = []
        for offset in range(scan_range):
            sx = int(round(mid[0] + offset * normal[0]))
            sy = int(round(mid[1] + offset * normal[1]))
            if 0 <= sx < w and 0 <= sy < h:
                intensities.append(gray_image[sy, sx])
            else:
                intensities.append(0)

        if len(intensities) < 5:
            continue

        # Find the point where intensity rises sharply (entering the bright panel)
        # We look for the steepest positive gradient (dark->bright transition)
        intensities = np.array(intensities, dtype=np.float64)
        gradients = np.diff(intensities)

        # Find the strongest rising edge (frame -> panel transition)
        if len(gradients) > 0:
            max_grad_idx = np.argmax(gradients)
            max_grad_val = gradients[max_grad_idx]

            # Only refine if there's a significant gradient
            if max_grad_val > 15:
                # Place boundary just after the transition (inside the bright area)
                refined_offsets[edge_idx] = float(max_grad_idx + 1)

    # Apply the refinement by shrinking the rectangle inward
    # Average the opposing edge offsets for width/height adjustment
    # Edges 0,2 affect one dimension; edges 1,3 affect the other
    avg_offset_02 = (refined_offsets[0] + refined_offsets[2]) / 2.0
    avg_offset_13 = (refined_offsets[1] + refined_offsets[3]) / 2.0

    # Shrink dimensions
    new_w = max(rw - 2 * avg_offset_02, rw * 0.9)
    new_h = max(rh - 2 * avg_offset_13, rh * 0.9)

    # Shift center based on asymmetric offsets
    angle_rad = np.deg2rad(angle)
    # Direction along width
    dx_w = np.cos(angle_rad)
    dy_w = np.sin(angle_rad)
    # Direction along height
    dx_h = -np.sin(angle_rad)
    dy_h = np.cos(angle_rad)

    offset_w = (refined_offsets[0] - refined_offsets[2]) / 2.0
    offset_h = (refined_offsets[1] - refined_offsets[3]) / 2.0

    new_cx = cx + offset_w * dx_w + offset_h * dx_h
    new_cy = cy + offset_w * dy_w + offset_h * dy_h

    return ((new_cx, new_cy), (new_w, new_h), angle)


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

    This is the main entry point. It detects the backlight panel boundary that
    tightly fits the actual INNER edge of the bright panel, excluding the frame.

    The algorithm prioritizes tight inner boundary detection:
    1. Uses high-threshold segmentation to isolate only the bright panel
    2. Applies edge-scan refinement to find the exact panel-to-frame transition
    3. Falls back to edge-based and adaptive threshold methods if needed

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

    # Step 2: Primary method - bright threshold isolation
    # This isolates only the actual bright panel, excluding the semi-bright frame
    rect, contour, binary_output = detect_by_bright_threshold(
        gray_image, preprocessed,
        min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
    )

    # Step 3: Refine the detected rectangle using edge scanning
    # This pulls the boundary inward to the exact panel-frame transition
    if rect is not None:
        rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=25)

    # Step 4: If bright threshold failed, try edge-based detection
    if rect is None:
        rect, contour, edge_img = detect_by_edge(
            gray_image, preprocessed,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
        )
        if rect is not None:
            rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=25)
        if binary_output is None:
            binary_output = edge_img

    # Step 5: Fallback - Otsu threshold with erosion
    if rect is None:
        _, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Erode to tighten boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=2)
        cleaned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=1)

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

        if rect is not None:
            rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=25)

    # Step 6: Fallback - adaptive threshold
    if rect is None:
        binary = cv2.adaptiveThreshold(
            preprocessed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c_offset,
        )
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
