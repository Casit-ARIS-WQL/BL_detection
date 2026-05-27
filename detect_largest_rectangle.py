"""
Backlight Panel Rectangle Detection via Mask Expansion

This module detects the backlight panel rectangle in a grayscale image by
expanding a mask outward from the image center. Each mask pixel stops expanding
when it encounters a point with a significantly different grayscale value from
the center. Once all frontier pixels have stopped, the minimum inscribed
rectangle of the mask is computed and visualized.

Algorithm:
1. Start from the image center pixel and record its grayscale value.
2. Use BFS-like expansion: each frontier pixel expands to its neighbors.
3. A pixel stops expanding if the neighbor's grayscale difference from the
   center exceeds a threshold.
4. Expansion ends when no more frontier pixels can grow.
5. Compute the minimum inscribed (axis-aligned) rectangle of the final mask.
6. Output the rectangle visualization.
"""

import cv2
import numpy as np


def preprocess_image(gray_image):
    """Apply preprocessing to reduce noise while preserving edges.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8).

    Returns:
        Preprocessed grayscale image with noise reduced but edges preserved.
    """
    denoised = cv2.bilateralFilter(gray_image, d=7, sigmaColor=50, sigmaSpace=50)
    return denoised


def expand_mask_from_center(gray_image, threshold=50):
    """Expand a mask from the image center, stopping at high-contrast boundaries.

    Starting from the center pixel, the mask grows outward using BFS. Each
    frontier pixel attempts to expand to its 4-connected neighbors. If a
    neighbor's grayscale value differs from the center value by more than
    the threshold, that direction stops. Expansion continues until all
    frontier pixels are blocked.

    Args:
        gray_image: Grayscale image (numpy array, dtype uint8).
        threshold: Maximum allowed grayscale difference from center value.
            Pixels exceeding this difference are treated as boundaries.

    Returns:
        Binary mask (numpy array, dtype uint8, same size as input) where
        255 indicates the expanded region and 0 indicates background.
    """
    h, w = gray_image.shape[:2]
    center_y, center_x = h // 2, w // 2
    center_value = int(gray_image[center_y, center_x])

    # Use OpenCV floodFill for efficient expansion from center.
    # floodFill checks each pixel's difference from its neighbor AND from the
    # seed value when using FLOODFILL_FIXED_RANGE flag, which matches our
    # requirement of stopping when grayscale differs from center by > threshold.
    #
    # The mask for floodFill must be 2 pixels larger than the image in each dimension.
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # FLOODFILL_FIXED_RANGE: compare each pixel to the seed pixel value
    # (center), not to its neighbor. This implements the "difference from
    # center" logic.
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

    # loDiff and upDiff define the acceptable range: center_value - threshold
    # to center_value + threshold
    cv2.floodFill(
        gray_image.copy(),  # work on a copy to avoid modifying input
        flood_mask,
        seedPoint=(center_x, center_y),
        newVal=255,
        loDiff=(threshold,),
        upDiff=(threshold,),
        flags=flags,
    )

    # Extract the actual mask (remove the 1-pixel border added by floodFill)
    # floodFill sets mask pixels to the value specified in flags bits 8-15,
    # but with uint8 mask it may just set to 1. Convert any nonzero to 255.
    raw_mask = flood_mask[1:-1, 1:-1]
    mask = np.where(raw_mask > 0, np.uint8(255), np.uint8(0))

    return mask


def find_max_inscribed_rect(mask):
    """Find the maximum axis-aligned inscribed rectangle within a binary mask.

    Uses a histogram-based approach (largest rectangle in histogram) applied
    row by row to find the largest axis-aligned rectangle fully contained
    within the mask region.

    Args:
        mask: Binary mask (numpy array, dtype uint8) where 255 = region.

    Returns:
        Tuple (x, y, w, h) of the inscribed rectangle, or None if not found.
    """
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.int32)

    # Build height histogram using vectorized operations
    heights = np.zeros((h, w), dtype=np.int32)
    heights[0] = binary[0]
    for row in range(1, h):
        heights[row] = np.where(binary[row] == 1, heights[row - 1] + 1, 0)

    # For each row, find the largest rectangle in the histogram
    best_area = 0
    best_rect = None  # (x, y, w, h)

    for row in range(h):
        hist = heights[row]
        stack = []  # stack of indices
        col = 0

        while col <= w:
            cur_height = int(hist[col]) if col < w else 0

            if not stack or cur_height >= int(hist[stack[-1]]):
                stack.append(col)
                col += 1
            else:
                top = stack.pop()
                top_height = int(hist[top])
                width = col if not stack else col - stack[-1] - 1
                area = top_height * width

                if area > best_area:
                    best_area = area
                    rect_h = top_height
                    rect_w = width
                    rect_x = stack[-1] + 1 if stack else 0
                    rect_y = row - rect_h + 1
                    best_rect = (rect_x, rect_y, rect_w, rect_h)

    return best_rect


def detect_largest_rectangle(
    gray_image,
    threshold=50,
    min_area_ratio=0.01,
    max_area_ratio=0.95,
    **kwargs,
):
    """Detect the largest rectangle in a grayscale image using mask expansion.

    Expands a mask from the image center outward. Each pixel stops expanding
    when it encounters a significantly different grayscale value. Then computes
    the maximum inscribed rectangle within the resulting mask.

    Args:
        gray_image: Input grayscale image (numpy array, dtype uint8, single channel).
        threshold: Grayscale difference threshold for stopping mask expansion.
        min_area_ratio: Minimum rectangle area as fraction of image area.
        max_area_ratio: Maximum rectangle area as fraction of image area.
        **kwargs: Additional keyword arguments (kept for API compatibility).

    Returns:
        A dictionary with the following keys:
            - "rect": ((center_x, center_y), (width, height), angle) or None
            - "box": numpy array of 4 corner points (int) or None
            - "contour": the detected contour or None
            - "binary": the final binary mask used for detection

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

    h, w = gray_image.shape[:2]
    image_area = h * w

    # Step 1: Preprocess to reduce noise
    preprocessed = preprocess_image(gray_image)

    # Step 2: Expand mask from center
    mask = expand_mask_from_center(preprocessed, threshold=threshold)

    # Step 3: Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: Find the maximum inscribed rectangle
    inscribed = find_max_inscribed_rect(mask)

    rect = None
    box = None
    contour = None

    if inscribed is not None:
        rx, ry, rw, rh = inscribed
        rect_area = rw * rh

        # Validate area constraints
        if rect_area >= image_area * min_area_ratio and rect_area <= image_area * max_area_ratio:
            # Build rect in OpenCV minAreaRect format: ((cx, cy), (w, h), angle)
            cx = rx + rw / 2.0
            cy = ry + rh / 2.0
            rect = ((cx, cy), (float(rw), float(rh)), 0.0)

            # Build box points (4 corners)
            box = np.array([
                [rx, ry],
                [rx + rw, ry],
                [rx + rw, ry + rh],
                [rx, ry + rh]
            ], dtype=np.intp)

            # Build contour
            contour = np.array([
                [[rx, ry]],
                [[rx + rw, ry]],
                [[rx + rw, ry + rh]],
                [[rx, ry + rh]]
            ], dtype=np.int32)

    return {
        "rect": rect,
        "box": box,
        "contour": contour,
        "binary": mask,
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

    # Detect using mask expansion from center
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

    # Also save the mask for visualization
    mask_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, result["binary"])
    print(f"Mask saved to '{mask_path}'")
