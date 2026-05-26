"""
Precise Backlight Panel Rectangle Detection Algorithm

This module detects the backlight panel rectangle in a grayscale image with
high precision, ensuring the detected boundary tightly fits the actual inner
panel edge (the bright illuminated area), not the outer frame.

Key techniques:
1. Multi-scale gradient analysis to locate the inner panel boundary
2. Scan-line based edge detection from center outward for each side
3. High-percentile thresholding with aggressive erosion
4. Hough line detection for straight edge refinement
5. Sub-pixel edge refinement using intensity gradient profiles
"""

import cv2
import numpy as np


def preprocess_image(gray_image):
    """Apply preprocessing to reduce noise while preserving edges.

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


def _find_bright_region_center(gray_image):
    """Find the center of the brightest region in the image.

    Uses moments of the thresholded bright region to find the centroid,
    which serves as a reliable starting point for outward scanning.

    Args:
        gray_image: Grayscale image.

    Returns:
        (cx, cy) center coordinates of the bright region.
    """
    # Use a very high threshold to find the core bright area
    thresh_val = int(np.percentile(gray_image, 90))
    thresh_val = max(thresh_val, 200)
    _, binary = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY)

    moments = cv2.moments(binary)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cy, cx = gray_image.shape[0] // 2, gray_image.shape[1] // 2

    return cx, cy


def _find_edge_by_gradient_scan(gray_image, start_x, start_y, dx, dy,
                                max_distance, num_scanlines=15,
                                scanline_spacing=10):
    """Find the panel inner edge by scanning outward from the bright center.

    Scans multiple parallel lines from the bright panel center outward,
    looking for where intensity first drops significantly below the center
    brightness. Uses a threshold-based approach combined with gradient
    detection for robustness.

    Args:
        gray_image: Grayscale image.
        start_x, start_y: Starting point (inside bright panel).
        dx, dy: Scan direction (outward from center).
        max_distance: Maximum scan distance in pixels.
        num_scanlines: Number of parallel scan lines.
        scanline_spacing: Spacing between parallel scan lines in pixels.

    Returns:
        Estimated edge position (distance from start along scan direction),
        or max_distance if no edge found.
    """
    h, w = gray_image.shape[:2]

    # Perpendicular direction for parallel scanlines
    perp_dx, perp_dy = -dy, dx

    # Determine center brightness level
    center_intensity = float(gray_image[
        max(0, min(h - 1, start_y)),
        max(0, min(w - 1, start_x))
    ])
    # Threshold: edge is where intensity drops below 70% of center brightness
    intensity_threshold = center_intensity * 0.70

    edge_positions = []

    for sl in range(num_scanlines):
        offset = (sl - num_scanlines // 2) * scanline_spacing
        sx = start_x + int(offset * perp_dx)
        sy = start_y + int(offset * perp_dy)

        # Sample intensities along this scanline
        intensities = []
        for d in range(max_distance):
            px = int(round(sx + d * dx))
            py = int(round(sy + d * dy))
            if 0 <= px < w and 0 <= py < h:
                intensities.append(float(gray_image[py, px]))
            else:
                break

        if len(intensities) < 10:
            continue

        intensities = np.array(intensities)

        # Smooth to reduce noise
        kernel_size = 5
        if len(intensities) > kernel_size:
            smoothed = np.convolve(intensities, np.ones(kernel_size) / kernel_size,
                                   mode='valid')
        else:
            smoothed = intensities

        # Method 1: Find where intensity drops below threshold (first crossing)
        # This gives the INNER edge of the panel
        threshold_pos = None
        for i in range(len(smoothed)):
            if smoothed[i] < intensity_threshold:
                threshold_pos = i + kernel_size // 2
                break

        # Method 2: Find the first significant negative gradient
        gradient = np.diff(smoothed)
        gradient_pos = None
        if len(gradient) >= 3:
            # Look for the first significant drop (not strongest - first!)
            drop_threshold = -max(10, center_intensity * 0.03)
            for i in range(len(gradient)):
                if gradient[i] < drop_threshold:
                    gradient_pos = i + kernel_size // 2
                    break

        # Use the minimum of both methods (most conservative = tightest boundary)
        candidates = [p for p in [threshold_pos, gradient_pos] if p is not None]
        if candidates:
            edge_positions.append(min(candidates))

    if len(edge_positions) >= 3:
        # Use median for robustness against outliers
        return int(np.median(edge_positions))
    elif len(edge_positions) > 0:
        return int(np.median(edge_positions))
    else:
        return max_distance


def detect_by_scanline(gray_image, preprocessed,
                       min_area_ratio=0.01, max_area_ratio=0.95):
    """Detect the inner panel rectangle by scanning outward from the bright center.

    This is the most accurate method: it starts from the known bright center
    of the panel and scans outward in 4 directions to find exactly where the
    brightness drops off (the inner panel edge).

    Args:
        gray_image: Original grayscale image.
        preprocessed: Preprocessed (denoised) grayscale image.
        min_area_ratio: Minimum area ratio.
        max_area_ratio: Maximum area ratio.

    Returns:
        A tuple (rect, contour, binary_image) or (None, None, None).
    """
    h, w = gray_image.shape[:2]
    image_area = h * w

    # Find the center of the bright panel
    cx, cy = _find_bright_region_center(preprocessed)

    # Determine maximum scan distances (half image dimensions)
    max_dist_x = min(cx, w - cx, w // 2)
    max_dist_y = min(cy, h - cy, h // 2)

    # Scan in 4 cardinal directions from center
    # Use more scanlines and larger spacing for better coverage
    num_sl = max(11, min(31, min(h, w) // 30))
    spacing = max(5, min(20, min(h, w) // 60))

    # Right edge
    right_dist = _find_edge_by_gradient_scan(
        preprocessed, cx, cy, 1, 0, max_dist_x,
        num_scanlines=num_sl, scanline_spacing=spacing)
    # Left edge
    left_dist = _find_edge_by_gradient_scan(
        preprocessed, cx, cy, -1, 0, max_dist_x,
        num_scanlines=num_sl, scanline_spacing=spacing)
    # Down edge
    down_dist = _find_edge_by_gradient_scan(
        preprocessed, cx, cy, 0, 1, max_dist_y,
        num_scanlines=num_sl, scanline_spacing=spacing)
    # Up edge
    up_dist = _find_edge_by_gradient_scan(
        preprocessed, cx, cy, 0, -1, max_dist_y,
        num_scanlines=num_sl, scanline_spacing=spacing)

    # Construct rectangle from edge distances
    x_left = cx - left_dist
    x_right = cx + right_dist
    y_top = cy - up_dist
    y_bottom = cy + down_dist

    # Validate the detected region
    det_w = x_right - x_left
    det_h = y_bottom - y_top
    det_area = det_w * det_h

    if det_area < image_area * min_area_ratio or det_area > image_area * max_area_ratio:
        return None, None, None

    if det_w < 20 or det_h < 20:
        return None, None, None

    # Check aspect ratio is reasonable
    aspect = max(det_w, det_h) / max(min(det_w, det_h), 1)
    if aspect > 10:
        return None, None, None

    # Create the rectangle (axis-aligned from scanline detection)
    rect_cx = (x_left + x_right) / 2.0
    rect_cy = (y_top + y_bottom) / 2.0
    rect = ((rect_cx, rect_cy), (float(det_w), float(det_h)), 0.0)

    # Create a contour for the detected region
    contour = np.array([
        [[x_left, y_top]],
        [[x_right, y_top]],
        [[x_right, y_bottom]],
        [[x_left, y_bottom]]
    ], dtype=np.int32)

    return rect, contour, None


def detect_by_bright_threshold(gray_image, preprocessed,
                               min_area_ratio=0.01, max_area_ratio=0.95):
    """Detect rectangle by isolating only the brightest region (inner panel).

    Uses a very high threshold to select only the actual illuminated panel area,
    then applies aggressive erosion to tighten the boundary to the true inner edge.

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

    # Use very high percentile to isolate only the truly bright panel
    high_val = np.percentile(preprocessed, 95)
    low_val = np.percentile(preprocessed, 50)

    # Set threshold high to exclude the semi-bright frame region
    threshold = int(low_val + 0.8 * (high_val - low_val))
    threshold = max(threshold, 200)

    _, binary = cv2.threshold(preprocessed, threshold, 255, cv2.THRESH_BINARY)

    # Apply stronger erosion to pull boundary inward away from frame
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.erode(binary, erode_kernel, iterations=2)

    # Close small gaps within the panel
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=3)

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
    lower = int(max(50, 0.8 * median_val))
    upper = int(min(255, 2.0 * median_val))

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


def refine_rect_with_edge_scan(gray_image, rect, scan_range=40, num_samples=20):
    """Refine rectangle edges by scanning for the steepest intensity transition.

    For each side of the rectangle, scans multiple points along the edge
    (not just the midpoint) and uses the median gradient position to determine
    where the actual panel-to-frame transition occurs.

    Args:
        gray_image: Original grayscale image.
        rect: The initial minAreaRect ((cx,cy), (w,h), angle).
        scan_range: Number of pixels to scan from each edge toward center.
        num_samples: Number of sample points along each edge.

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

    refined_offsets = [0.0, 0.0, 0.0, 0.0]

    for edge_idx in range(4):
        p1 = box[edge_idx]
        p2 = box[(edge_idx + 1) % 4]

        # Edge direction and inward normal
        edge_dir = p2 - p1
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1:
            continue

        edge_unit = edge_dir / edge_len
        # Inward normal (pointing toward rectangle center)
        normal = np.array([-edge_unit[1], edge_unit[0]])

        # Ensure normal points inward (toward center)
        mid = (p1 + p2) / 2.0
        to_center = np.array([cx - mid[0], cy - mid[1]])
        if np.dot(normal, to_center) < 0:
            normal = -normal

        # Sample multiple points along this edge
        edge_positions = []
        for sample_idx in range(num_samples):
            t = (sample_idx + 1) / (num_samples + 1)
            sample_pt = p1 + t * edge_dir

            # Scan inward from this edge point
            intensities = []
            for offset in range(scan_range):
                sx = int(round(sample_pt[0] + offset * normal[0]))
                sy = int(round(sample_pt[1] + offset * normal[1]))
                if 0 <= sx < w and 0 <= sy < h:
                    intensities.append(float(gray_image[sy, sx]))
                else:
                    intensities.append(0)

            if len(intensities) < 8:
                continue

            intensities = np.array(intensities)

            # Smooth the profile
            if len(intensities) >= 5:
                kernel = np.ones(3) / 3
                smoothed = np.convolve(intensities, kernel, mode='valid')
            else:
                smoothed = intensities

            gradient = np.diff(smoothed)
            if len(gradient) < 3:
                continue

            # Find the strongest rising edge (dark frame -> bright panel)
            max_grad_idx = np.argmax(gradient)
            max_grad_val = gradient[max_grad_idx]

            if max_grad_val > 10:
                # Edge position is where the transition happens
                edge_positions.append(max_grad_idx + 2)  # +2 for smoothing offset

        if len(edge_positions) >= 3:
            # Use median for robustness
            refined_offsets[edge_idx] = float(np.median(edge_positions))

    # Apply the refinement by shrinking the rectangle inward
    avg_offset_02 = (refined_offsets[0] + refined_offsets[2]) / 2.0
    avg_offset_13 = (refined_offsets[1] + refined_offsets[3]) / 2.0

    # Shrink dimensions (allow up to 15% shrink per side)
    max_shrink_w = rw * 0.15
    max_shrink_h = rh * 0.15
    shrink_w = min(avg_offset_02, max_shrink_w)
    shrink_h = min(avg_offset_13, max_shrink_h)

    new_w = rw - 2 * shrink_w
    new_h = rh - 2 * shrink_h

    # Shift center based on asymmetric offsets
    angle_rad = np.deg2rad(angle)
    dx_w = np.cos(angle_rad)
    dy_w = np.sin(angle_rad)
    dx_h = -np.sin(angle_rad)
    dy_h = np.cos(angle_rad)

    offset_w_diff = min(abs(refined_offsets[0] - refined_offsets[2]),
                        max_shrink_w) * np.sign(refined_offsets[0] - refined_offsets[2])
    offset_h_diff = min(abs(refined_offsets[1] - refined_offsets[3]),
                        max_shrink_h) * np.sign(refined_offsets[1] - refined_offsets[3])

    offset_w = offset_w_diff / 2.0
    offset_h = offset_h_diff / 2.0

    new_cx = cx + offset_w * dx_w + offset_h * dx_h
    new_cy = cy + offset_w * dy_w + offset_h * dy_h

    return ((new_cx, new_cy), (new_w, new_h), angle)


def _refine_with_hough_lines(gray_image, rect):
    """Refine rectangle using Hough line detection for straight edges.

    Detects straight lines near the initial rectangle boundaries and
    uses them to produce a more precise axis-aligned or rotated rectangle.

    Args:
        gray_image: Grayscale image.
        rect: Initial rectangle estimate ((cx,cy), (w,h), angle).

    Returns:
        Refined rect or the original if refinement fails.
    """
    if rect is None:
        return rect

    center, size, angle = rect
    cx, cy = center
    rw, rh = size
    h, w = gray_image.shape[:2]

    # Create a mask around the rectangle boundary (ROI for line detection)
    margin = int(max(rw, rh) * 0.1)
    box = cv2.boxPoints(rect)

    # Create edge image in the region of interest
    x_min = max(0, int(box[:, 0].min()) - margin)
    x_max = min(w, int(box[:, 0].max()) + margin)
    y_min = max(0, int(box[:, 1].min()) - margin)
    y_max = min(h, int(box[:, 1].max()) + margin)

    roi = gray_image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return rect

    # Edge detection on ROI
    edges = cv2.Canny(roi, 80, 200, apertureSize=3, L2gradient=True)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min(rw, rh) * 0.3,
                            maxLineGap=20)

    if lines is None or len(lines) < 2:
        return rect

    # Classify lines as roughly horizontal or vertical
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_angle = abs(np.arctan2(y2 - y1, x2 - x1))
        if line_angle < np.pi / 6 or line_angle > 5 * np.pi / 6:
            # Horizontal-ish
            h_lines.append((y1 + y_min + y2 + y_min) / 2.0)
        elif np.pi / 3 < line_angle < 2 * np.pi / 3:
            # Vertical-ish
            v_lines.append((x1 + x_min + x2 + x_min) / 2.0)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return rect

    # Find the inner-most horizontal and vertical lines
    h_lines = sorted(h_lines)
    v_lines = sorted(v_lines)

    # Inner boundaries: closest to center
    h_top = [y for y in h_lines if y < cy]
    h_bottom = [y for y in h_lines if y > cy]
    v_left = [x for x in v_lines if x < cx]
    v_right = [x for x in v_lines if x > cx]

    if not h_top or not h_bottom or not v_left or not v_right:
        return rect

    # Take the innermost lines (closest to center)
    top = max(h_top)
    bottom = min(h_bottom)
    left = max(v_left)
    right = min(v_right)

    new_w = right - left
    new_h = bottom - top

    if new_w < rw * 0.5 or new_h < rh * 0.5:
        return rect
    if new_w > rw * 1.1 or new_h > rh * 1.1:
        return rect

    new_cx = (left + right) / 2.0
    new_cy = (top + bottom) / 2.0

    return ((new_cx, new_cy), (new_w, new_h), 0.0)


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

    The algorithm uses a multi-strategy approach prioritizing inner boundary:
    1. Scanline-based detection from bright center outward (most accurate)
    2. High-threshold segmentation with aggressive erosion
    3. Edge-scan refinement to find exact panel-to-frame transition
    4. Hough line detection for straight edge refinement
    5. Falls back to edge-based and adaptive threshold methods if needed

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

    # Step 2: Primary method - scanline-based detection from center outward
    # This is the most accurate method for finding the inner panel edge
    rect, contour, _ = detect_by_scanline(
        gray_image, preprocessed,
        min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
    )

    # Step 3: If scanline worked, refine with Hough lines for straight edges
    if rect is not None:
        rect = _refine_with_hough_lines(preprocessed, rect)

    # Step 4: If scanline failed, try bright threshold isolation
    if rect is None:
        rect, contour, binary_output = detect_by_bright_threshold(
            gray_image, preprocessed,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
        )
        # Refine the detected rectangle using multi-sample edge scanning
        if rect is not None:
            rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=40,
                                              num_samples=20)

    # Step 5: If bright threshold failed, try edge-based detection
    if rect is None:
        rect, contour, edge_img = detect_by_edge(
            gray_image, preprocessed,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio
        )
        if rect is not None:
            rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=40,
                                              num_samples=20)
        if binary_output is None:
            binary_output = edge_img

    # Step 6: Fallback - Otsu threshold with strong erosion
    if rect is None and use_otsu_fallback:
        _, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Stronger erosion to tighten boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(binary, kernel, iterations=3)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=2)

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
            rect = refine_rect_with_edge_scan(gray_image, rect, scan_range=40,
                                              num_samples=20)

    # Step 7: Fallback - adaptive threshold
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
