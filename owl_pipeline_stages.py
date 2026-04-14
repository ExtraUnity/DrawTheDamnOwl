from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from owl_pipeline_utils import (
    CANVAS_SIZE,
    mask_bbox,
    overlay_dense_gray,
    overlay_sparse_gray,
    serialize_contour,
    serialize_contours,
)


# Stage 00 parameters (kept aligned with existing implementation)
HEAD_SPLIT_RATIO = 0.42
MIN_POINTS_FOR_ELLIPSE = 20
ELLIPSE_THICKNESS = 2

# Stage 01 parameters (kept aligned with existing implementation)
OUTER_CONTOUR_THICKNESS = 2
SIMPLIFY_EPSILON_RATIO = 0.005

# Stage 02 readability constants
STAGE2_HEAD_RATIO = 0.42
STAGE2_EYE_SEP_FROM_HEAD_W = 0.46
STAGE2_EYE_Y_FROM_HEAD_H = 0.16
STAGE2_EYE_SEARCH_RADIUS = 10


def _require_bbox(mask: np.ndarray, stage_name: str) -> Tuple[int, int, int, int]:
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError(f"Empty mask for {stage_name}")
    return bbox


def points_from_mask_region(mask_region: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    ys, xs = np.where(mask_region > 0)
    if len(xs) == 0:
        return np.empty((0, 1, 2), dtype=np.int32)

    pts = np.stack([xs + offset_x, ys + offset_y], axis=1).astype(np.int32)
    return pts.reshape(-1, 1, 2)


def fit_ellipse_from_points(points: np.ndarray) -> Optional[Dict[str, float]]:
    if len(points) < 5:
        return None

    ellipse = cv2.fitEllipse(points)
    (cx, cy), (w, h), angle = ellipse

    # Keep axis order stable for annotation consumers.
    if h > w:
        w, h = h, w
        angle = (angle + 90.0) % 180.0

    return {
        "cx": float(cx),
        "cy": float(cy),
        "width": float(w),
        "height": float(h),
        "angle": float(angle),
    }


def ellipse_from_region(mask: np.ndarray, y0: int, y1: int) -> Optional[Dict[str, float]]:
    region = mask[y0:y1, :]
    points = points_from_mask_region(region, offset_x=0, offset_y=y0)
    if len(points) < MIN_POINTS_FOR_ELLIPSE:
        return None
    return fit_ellipse_from_points(points)


def fallback_body_ellipse(mask: np.ndarray) -> Optional[Dict[str, float]]:
    points = points_from_mask_region(mask)
    if len(points) < MIN_POINTS_FOR_ELLIPSE:
        return None
    return fit_ellipse_from_points(points)


def derive_head_and_body_ellipses(mask: np.ndarray) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None

    x_min, y_min, x_max, y_max = bbox
    owl_h = y_max - y_min + 1

    split_y = y_min + int(round(owl_h * HEAD_SPLIT_RATIO))
    split_y = max(y_min + 5, min(split_y, y_max - 5))

    head_ellipse = ellipse_from_region(mask, y_min, split_y)
    body_ellipse = ellipse_from_region(mask, split_y, y_max + 1)

    full_ellipse = fallback_body_ellipse(mask)
    if body_ellipse is None:
        body_ellipse = full_ellipse

    if head_ellipse is None and full_ellipse is not None:
        cx = (x_min + x_max) / 2.0
        cy = y_min + 0.28 * owl_h
        width = 0.45 * (x_max - x_min + 1)
        height = 0.32 * owl_h
        head_ellipse = {
            "cx": float(cx),
            "cy": float(cy),
            "width": float(width),
            "height": float(height),
            "angle": 0.0,
        }

    return head_ellipse, body_ellipse


def draw_ellipse(canvas: np.ndarray, ellipse: Optional[Dict[str, float]], color: int = 255, thickness: int = 2) -> None:
    if ellipse is None:
        return

    center = (int(round(ellipse["cx"])), int(round(ellipse["cy"])))
    axes = (
        max(1, int(round(ellipse["width"] / 2.0))),
        max(1, int(round(ellipse["height"] / 2.0))),
    )
    angle = float(ellipse["angle"])
    cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness)


def interior_mask_by_distance(mask: np.ndarray, min_distance_px: float) -> np.ndarray:
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return np.where(dist >= float(min_distance_px), 255, 0).astype(np.uint8)


def stage_00_base_ellipses(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    head_ellipse, body_ellipse = derive_head_and_body_ellipses(mask)
    if head_ellipse is None and body_ellipse is None:
        raise ValueError("No valid owl shape found in mask for stage 00")

    layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    draw_ellipse(layer, body_ellipse, thickness=ELLIPSE_THICKNESS)
    draw_ellipse(layer, head_ellipse, thickness=ELLIPSE_THICKNESS)

    meta = {
        "head_ellipse": head_ellipse,
        "body_ellipse": body_ellipse,
        "head_split_ratio": HEAD_SPLIT_RATIO,
        "ellipse_thickness": ELLIPSE_THICKNESS,
    }
    return layer, layer.copy(), meta


def find_main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def simplify_contour(contour: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_ratio * perimeter
    return cv2.approxPolyDP(contour, epsilon, closed=True)


def stage_01_outer_contour(mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    contour = find_main_contour(mask)
    if contour is None or len(contour) < 5:
        raise ValueError("No valid contour found for stage 01")

    simplified = simplify_contour(contour, SIMPLIFY_EPSILON_RATIO)
    layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    cv2.drawContours(layer, [simplified], -1, 255, OUTER_CONTOUR_THICKNESS)
    cumulative = overlay_sparse_gray(prev_stage, layer)

    meta = {
        "simplify_epsilon_ratio": SIMPLIFY_EPSILON_RATIO,
        "contour_thickness": OUTER_CONTOUR_THICKNESS,
        "num_contour_points": int(len(simplified)),
        "outer_contour": serialize_contour(simplified),
    }
    return layer, cumulative, meta


def _stage2_head_prior(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
    x_min, y_min, x_max, y_max = bbox
    owl_w = x_max - x_min + 1
    owl_h = y_max - y_min + 1

    head_ellipse, _ = derive_head_and_body_ellipses(mask)
    if head_ellipse is None:
        center_x = 0.5 * (x_min + x_max)
        center_y = y_min + 0.20 * owl_h
        head_w = 0.55 * owl_w
        head_h = 0.32 * owl_h
    else:
        center_x = float(head_ellipse["cx"])
        center_y = float(head_ellipse["cy"])
        head_w = float(head_ellipse["width"])
        head_h = float(head_ellipse["height"])

    eye_sep = STAGE2_EYE_SEP_FROM_HEAD_W * head_w
    return {
        "left_x": center_x - 0.5 * eye_sep,
        "right_x": center_x + 0.5 * eye_sep,
        "eye_y": center_y - STAGE2_EYE_Y_FROM_HEAD_H * head_h,
        "eye_sep": eye_sep,
        "head_w": head_w,
    }


def _stage2_head_mask(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    owl_h = y_max - y_min + 1
    y_head_max = min(y_max, y_min + int(round(STAGE2_HEAD_RATIO * owl_h)))
    head_mask = np.zeros_like(mask)
    head_mask[y_min:y_head_max + 1, x_min:x_max + 1] = mask[y_min:y_head_max + 1, x_min:x_max + 1]
    return head_mask


def _stage2_pick_dark_point(gray: np.ndarray, valid_mask: np.ndarray, anchor_x: float, anchor_y: float) -> Tuple[float, float]:
    sr = STAGE2_EYE_SEARCH_RADIUS
    ax = int(round(anchor_x))
    ay = int(round(anchor_y))

    y0 = max(0, ay - sr)
    y1 = min(gray.shape[0], ay + sr + 1)
    x0 = max(0, ax - sr)
    x1 = min(gray.shape[1], ax + sr + 1)
    if y1 <= y0 or x1 <= x0:
        return float(anchor_x), float(anchor_y)

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    valid = valid_mask[y0:y1, x0:x1] > 0
    if not np.any(valid):
        return float(anchor_x), float(anchor_y)

    yy, xx = np.indices(patch.shape)
    gx = xx.astype(np.float32) + x0
    gy = yy.astype(np.float32) + y0
    dist = np.sqrt((gx - float(anchor_x)) ** 2 + (gy - float(anchor_y)) ** 2)
    # Bias toward darkness, with a soft distance penalty to keep anatomy stable.
    score = (255.0 - patch) - 1.9 * dist
    score[~valid] = -1e9

    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x0 + px), float(y0 + py)


def _stage2_draw_features(
    canvas_shape: Tuple[int, int],
    left_x: float,
    right_x: float,
    eye_y: float,
    eye_radius: float,
    beak_x: float,
    beak_y: float,
) -> np.ndarray:
    layer = np.zeros(canvas_shape, dtype=np.uint8)
    sep = max(1.0, right_x - left_x)

    eye_r = int(round(np.clip(0.70 * eye_radius, 3.0, min(10.0, max(3.0, 0.35 * sep)))))
    for ex in [left_x, right_x]:
        c = (int(round(ex)), int(round(eye_y)))
        cv2.circle(layer, c, eye_r, 255, 2)
        cv2.circle(layer, c, max(1, eye_r // 3), 255, -1)

    mid_x = 0.5 * (left_x + right_x)
    base_y = eye_y + 0.30 * sep
    beak_w = max(8.0, 0.38 * sep)
    pts = np.array(
        [
            [int(round(mid_x - 0.5 * beak_w)), int(round(base_y))],
            [int(round(beak_x)), int(round(max(beak_y, base_y + 8)))],
            [int(round(mid_x + 0.5 * beak_w)), int(round(base_y))],
        ],
        dtype=np.int32,
    )
    cv2.polylines(layer, [pts], isClosed=True, color=255, thickness=2)
    return layer


def stage_02_facial_features(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox = _require_bbox(mask, "stage 02")

    x_min, y_min, x_max, y_max = bbox
    owl_w = x_max - x_min + 1
    owl_h = y_max - y_min + 1

    prior = _stage2_head_prior(mask, bbox)
    left_anchor_x = float(prior["left_x"])
    right_anchor_x = float(prior["right_x"])
    eye_anchor_y = float(prior["eye_y"])
    eye_sep_prior = float(prior["eye_sep"])
    head_w = float(prior["head_w"])

    head_mask = _stage2_head_mask(mask, bbox)
    left_x, left_y = _stage2_pick_dark_point(gray, head_mask, left_anchor_x, eye_anchor_y)
    right_x, right_y = _stage2_pick_dark_point(gray, head_mask, right_anchor_x, eye_anchor_y)
    if left_x > right_x:
        left_x, right_x = right_x, left_x
        left_y, right_y = right_y, left_y

    x_slack = 0.18 * eye_sep_prior
    left_x = float(np.clip(0.70 * left_anchor_x + 0.30 * left_x, left_anchor_x - x_slack, left_anchor_x + x_slack))
    right_x = float(np.clip(0.70 * right_anchor_x + 0.30 * right_x, right_anchor_x - x_slack, right_anchor_x + x_slack))
    eye_y = float(0.65 * eye_anchor_y + 0.35 * (0.5 * (left_y + right_y)))
    eye_y = float(np.clip(eye_y, y_min + 0.08 * owl_h, y_min + 0.34 * owl_h))

    eye_radius = float(np.clip(0.10 * head_w, 4.0, 10.0))
    sep = right_x - left_x
    mode = "anchor_dark_spot"
    used_fallback = False
    if sep < 0.14 * owl_w or sep > 0.66 * owl_w or eye_y > y_min + 0.36 * owl_h:
        sep = 0.24 * owl_w
        left_x = float(0.5 * (x_min + x_max) - 0.5 * sep)
        right_x = float(0.5 * (x_min + x_max) + 0.5 * sep)
        eye_y = float(y_min + 0.24 * owl_h)
        eye_radius = 6.0
        mode = "fallback"
        used_fallback = True

    # Beak tip from dark response in a narrow window below eyes.
    mid_x = 0.5 * (left_x + right_x)
    x1 = max(x_min, int(round(mid_x - 0.32 * sep)))
    x2 = min(x_max, int(round(mid_x + 0.32 * sep)))
    y1 = max(y_min, int(round(eye_y + 0.10 * sep)))
    y2 = min(y_max, int(round(eye_y + 0.75 * sep)))
    beak_x = float(mid_x)
    beak_y = float(eye_y + 0.35 * sep)
    if x2 > x1 and y2 > y1:
        roi = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
        valid = mask[y1:y2 + 1, x1:x2 + 1] > 0
        if np.any(valid):
            yy, xx = np.indices(roi.shape)
            cx = mid_x - x1
            center_bias = 1.0 - np.clip(np.abs(xx.astype(np.float32) - cx) / max(1.0, 0.5 * (x2 - x1)), 0.0, 1.0)
            score = (255.0 - roi) + 18.0 * center_bias
            score[~valid] = -1e9
            py, px = np.unravel_index(int(np.argmax(score)), score.shape)
            beak_x = float(x1 + px)
            beak_y = float(max(y1 + py, eye_y + 2.0))

    layer = _stage2_draw_features(gray.shape, left_x, right_x, eye_y, eye_radius, beak_x, beak_y)

    area = int(round(np.pi * eye_radius * eye_radius))
    cumulative = overlay_sparse_gray(prev_stage, layer)
    meta = {
        "method": "silhouette_anchor_dark_spot",
        "eye_detection_mode": mode,
        "used_fallback_eye_pair": used_fallback,
        "eye_anchor_prior": {
            "left_x": float(left_anchor_x),
            "right_x": float(right_anchor_x),
            "y": float(eye_anchor_y),
            "sep": float(eye_sep_prior),
        },
        "left_eye": {"x": float(left_x), "y": float(eye_y), "area": area, "radius": float(eye_radius)},
        "right_eye": {"x": float(right_x), "y": float(eye_y), "area": area, "radius": float(eye_radius)},
        "beak_tip": {"x": float(beak_x), "y": float(beak_y)},
    }
    return layer, cumulative, meta


def _remove_outer_edge_band(edge_map: np.ndarray, mask: np.ndarray, width: int = 3) -> np.ndarray:
    contour = find_main_contour(mask)
    out = edge_map.copy()
    if contour is not None:
        cv2.drawContours(out, [contour], -1, 0, width)
    return out


def _row_bounds(mask: np.ndarray, y: int) -> Optional[Tuple[int, int]]:
    xs = np.where(mask[y] > 0)[0]
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max())


def _quadratic_curve(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], num_points: int = 24) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, num_points)
    pts = []
    for t in ts:
        x = (1.0 - t) * (1.0 - t) * p0[0] + 2.0 * (1.0 - t) * t * p1[0] + t * t * p2[0]
        y = (1.0 - t) * (1.0 - t) * p0[1] + 2.0 * (1.0 - t) * t * p1[1] + t * t * p2[1]
        pts.append([int(round(x)), int(round(y))])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def stage_03_part_boundaries(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    bbox = _require_bbox(mask, "stage 03")

    x_min, y_min, x_max, y_max = bbox
    owl_h = y_max - y_min + 1

    split_y = y_min + int(round(0.42 * owl_h))
    split_y = max(y_min + 8, min(split_y, y_max - 20))

    split_bounds = _row_bounds(mask, split_y)
    if split_bounds is None:
        raise ValueError("Could not derive split-row bounds for stage 03")

    left_split, right_split = split_bounds
    center_split = 0.5 * (left_split + right_split)
    center_bottom = 0.5 * (x_min + x_max)
    y_top = split_y + 6
    y_bottom = y_max - int(round(0.16 * owl_h))

    selected: List[np.ndarray] = []

    # Head/body separator as a mild downward arc.
    head_body = _quadratic_curve(
        (left_split + 2, split_y),
        (center_split, split_y + int(round(0.08 * owl_h))),
        (right_split - 2, split_y),
        num_points=30,
    )
    selected.append(head_body)

    # Three principal internal boundaries: left wing hint, chest center, right wing hint.
    left_wing = _quadratic_curve(
        (left_split + 0.28 * (center_split - left_split), y_top),
        (left_split + 0.20 * (center_split - left_split), 0.5 * (y_top + y_bottom)),
        (x_min + 0.38 * (center_bottom - x_min), y_bottom),
        num_points=24,
    )
    chest_sep = _quadratic_curve(
        (center_split, y_top + 3),
        (center_split, 0.5 * (y_top + y_bottom)),
        (center_bottom, y_bottom),
        num_points=24,
    )
    right_wing = _quadratic_curve(
        (right_split - 0.28 * (right_split - center_split), y_top),
        (right_split - 0.20 * (right_split - center_split), 0.5 * (y_top + y_bottom)),
        (x_max - 0.38 * (x_max - center_bottom), y_bottom),
        num_points=24,
    )
    selected.extend([left_wing, chest_sep, right_wing])

    # Tail/feet hint as a small V near the lower body.
    by = y_max - 2
    tail_hint = np.array(
        [[int(round(center_bottom - 10)), by - 8], [int(round(center_bottom)), by], [int(round(center_bottom + 10)), by - 8]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    selected.append(tail_hint)

    layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    if selected:
        for contour in selected:
            cv2.polylines(layer, [contour], isClosed=False, color=255, thickness=2)
    cumulative = overlay_sparse_gray(prev_stage, layer)

    meta = {
        "method": "semantic_geometry_guided_boundaries",
        "split_y": int(split_y),
        "num_part_boundaries": len(selected),
        "boundary_labels": ["head_body_separator", "left_wing_boundary", "chest_separator", "right_wing_boundary", "tail_feet_hint"],
        "part_boundaries": serialize_contours(selected),
    }
    return layer, cumulative, meta


def stage_04_inner_contours(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=20)

    edges = cv2.Canny(smooth, 18, 62)
    edges = cv2.bitwise_and(edges, mask)
    edges = _remove_outer_edge_band(edges, mask, width=4)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(edges)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 4 <= area <= 2200:
            cleaned[labels == label] = 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    selected: List[np.ndarray] = []
    for contour in sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True):
        peri = cv2.arcLength(contour, closed=False)
        if peri < 10.0 or peri > 520.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.010 * peri, closed=False)
        if len(approx) < 3:
            continue
        selected.append(approx)
        if len(selected) >= 85:
            break

    layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    if selected:
        cv2.drawContours(layer, selected, -1, 255, 1)
    cumulative = overlay_sparse_gray(prev_stage, layer)

    meta = {
        "method": "masked_midscale_contours",
        "num_inner_contours": len(selected),
        "inner_contours": serialize_contours(selected),
    }
    return layer, cumulative, meta


def stage_05_value_regions(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    inside = blur[mask > 0]
    if inside.size == 0:
        raise ValueError("Empty mask for stage 05")

    q1, q2, q3 = [int(v) for v in np.percentile(inside, [25, 50, 75])]
    bins = [q1, q2, q3]
    values = [int(v) for v in np.percentile(inside, [20, 40, 60, 80])]
    #values = [70, 100, 140, 190]
    layer = np.zeros_like(gray)

    masked_blur = blur.copy()
    masked_blur[mask == 0] = 0
    layer[(masked_blur > 0) & (masked_blur <= bins[0]) & (mask > 0)] = values[0]
    layer[(masked_blur > bins[0]) & (masked_blur <= bins[1]) & (mask > 0)] = values[1]
    layer[(masked_blur > bins[1]) & (masked_blur <= bins[2]) & (mask > 0)] = values[2]
    layer[(masked_blur > bins[2]) & (mask > 0)] = values[3]

    layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    edge_safe_mask = interior_mask_by_distance(mask, min_distance_px=4.0)
    layer = cv2.bitwise_and(layer, edge_safe_mask)
    cumulative = overlay_dense_gray(prev_stage, layer)

    meta = {
        "method": "masked_grayscale_quantization",
        "quantile_thresholds": bins,
        "output_values": values,
        "edge_safe_mask": "distance_transform_min4px",
    }
    return layer, cumulative, meta


def stage_06_feather_masses(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source = prev_stage.copy() if prev_stage is not None else gray
    source = cv2.GaussianBlur(source, (5, 5), 0)

    bbox = _require_bbox(mask, "stage 06")
    x_min, y_min, x_max, y_max = bbox
    owl_h = y_max - y_min + 1

    # Focus feather masses on owl body and suppress feet/perch structures.
    body_bottom = y_min + int(round(0.80 * owl_h))
    body_bottom = max(y_min + 12, min(body_bottom, y_max))
    body_mask = np.zeros_like(mask)
    body_mask[y_min:body_bottom + 1, :] = mask[y_min:body_bottom + 1, :]
    if np.count_nonzero(body_mask) == 0:
        body_mask = mask.copy()

    inside_vals = source[body_mask > 0]
    if inside_vals.size == 0:
        raise ValueError("Empty body mask for stage 06")

    # Use stage-5 tonal structure to build broad, meaningful feather masses.
    samples = inside_vals.reshape(-1, 1).astype(np.float32)
    k = 5 if samples.shape[0] >= 500 else 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.7)
    _, labels_k, centers_k = cv2.kmeans(samples, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

    centers_flat = centers_k.reshape(-1)
    order = np.argsort(centers_flat)
    keep_ids = order[: max(2, k - 2)]  # keep darker/mid tones, drop brightest highlights.

    label_map = np.full(body_mask.shape, -1, dtype=np.int32)
    label_map[body_mask > 0] = labels_k.reshape(-1)

    coarse = np.zeros_like(body_mask)
    for cid in keep_ids:
        coarse[label_map == int(cid)] = 255

    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    coarse = cv2.bitwise_and(coarse, body_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((coarse > 0).astype(np.uint8), connectivity=8)
    selected_regions: List[np.ndarray] = []
    body_pixels = int(np.count_nonzero(body_mask))
    min_region_area = max(45, int(round(0.003 * body_pixels)))
    max_region_area = int(round(0.85 * body_pixels))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_region_area or area > max_region_area:
            continue
        region = np.zeros_like(coarse)
        region[labels == label] = 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.010 * cv2.arcLength(contour, True), closed=True)
        if len(approx) < 4:
            continue
        selected_regions.append(approx)

    # Fallback: if clustering over-prunes, use adaptive low/mid tone threshold in body mask.
    if len(selected_regions) == 0:
        tone_thr = int(np.percentile(inside_vals, 72))
        fallback = np.where((source <= tone_thr) & (body_mask > 0), 255, 0).astype(np.uint8)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        contours_fb, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours_fb, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_region_area or area > max_region_area:
                continue
            approx = cv2.approxPolyDP(contour, 0.010 * cv2.arcLength(contour, True), closed=True)
            if len(approx) < 4:
                continue
            selected_regions.append(approx)

    selected_regions = sorted(selected_regions, key=cv2.contourArea, reverse=True)[:28]

    layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    if selected_regions:
        cv2.drawContours(layer, selected_regions, -1, 125, -1)
    # Keep tonal masses away from the silhouette edge to avoid halo/rim artifacts in stage 7.
    edge_safe_body_mask = interior_mask_by_distance(body_mask, min_distance_px=6.0)
    layer = cv2.bitwise_and(layer, edge_safe_body_mask)

    cumulative = overlay_dense_gray(prev_stage, layer)

    meta = {
        "method": "body_focused_tonal_mass_grouping",
        "body_bottom": int(body_bottom),
        "kmeans_centers": [float(v) for v in centers_flat.tolist()],
        "num_regions": len(selected_regions),
        "edge_safe_mask": "distance_transform_min6px",
        "regions": serialize_contours(selected_regions),
    }
    return layer, cumulative, meta


def stage_07_fine_texture(image: np.ndarray, mask: np.ndarray, prev_stage: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)

    residual = cv2.subtract(gray, smooth)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)

    # Avoid boundary halo artifacts: keep fine texture well inside the owl silhouette.
    edge_safe_mask = interior_mask_by_distance(mask, min_distance_px=6.0)
    residual = cv2.bitwise_and(residual, edge_safe_mask)
    residual[residual < 18] = 0

    layer = residual
    if prev_stage is None:
        cumulative = layer.copy()
    else:
        cumulative = prev_stage.copy()
        # Max-overlay suppresses fine texture against stage 06 tonal fills; add a boosted texture residual instead.
        boosted = cv2.convertScaleAbs(layer, alpha=1.8, beta=0)
        cumulative = cv2.add(cumulative, boosted)
        cumulative = cv2.bitwise_and(cumulative, mask)

    meta = {
        "method": "high_frequency_residual",
        "smooth_sigma": 2.2,
        "residual_floor": 18,
        "edge_safe_mask": "distance_transform_min6px",
    }
    return layer, cumulative, meta


def stage_08_color(image: np.ndarray, mask: np.ndarray, prev_stage_gray: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    smooth = cv2.bilateralFilter(image, d=9, sigmaColor=45, sigmaSpace=30)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l_orig, a_chan, b_chan = cv2.split(lab)

    neutral = np.full_like(l_orig, 128, dtype=np.uint8)
    color_only_lab = cv2.merge([neutral, a_chan, b_chan])
    color_only = cv2.cvtColor(color_only_lab, cv2.COLOR_LAB2BGR)
    color_layer = np.zeros_like(image)
    color_layer[mask > 0] = color_only[mask > 0]

    if prev_stage_gray is None:
        prev_l = l_orig.copy()
    else:
        prev_l = prev_stage_gray.copy()

    cumulative_lab = cv2.merge([prev_l, a_chan, b_chan])
    cumulative = cv2.cvtColor(cumulative_lab, cv2.COLOR_LAB2BGR)
    cumulative_masked = np.zeros_like(cumulative)
    cumulative_masked[mask > 0] = cumulative[mask > 0]

    meta = {
        "method": "lab_chroma_transfer",
        "mean_a": float(np.mean(a_chan[mask > 0])) if np.any(mask > 0) else 0.0,
        "mean_b": float(np.mean(b_chan[mask > 0])) if np.any(mask > 0) else 0.0,
    }
    return color_layer, cumulative_masked, meta


def stage_09_background(image: np.ndarray, mask: np.ndarray, prev_stage_color: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    inv_mask = cv2.bitwise_not(mask)
    bg_layer = cv2.bitwise_and(image, image, mask=inv_mask)

    cumulative = np.zeros_like(image)
    if prev_stage_color is not None:
        cumulative = prev_stage_color.copy()

    # Fill a thin inner edge band from the source image to avoid dark/black halo artifacts.
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    edge_band = cv2.subtract(mask, inner)
    cumulative[edge_band > 0] = image[edge_band > 0]

    # Additional cleanup: if stylization left very dark pixels near the silhouette, restore them from source.
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    near_edge = (mask > 0) & (dist <= 10.0)
    cumulative_gray = cv2.cvtColor(cumulative, cv2.COLOR_BGR2GRAY)
    dark_near_edge = near_edge & (cumulative_gray < 45)
    cumulative[dark_near_edge] = image[dark_near_edge]

    cumulative[inv_mask > 0] = image[inv_mask > 0]

    meta = {
        "method": "outside_mask_background_transfer",
        "background_pixel_count": int(np.sum(inv_mask > 0)),
        "edge_band_filled_px": int(np.sum(edge_band > 0)),
        "dark_near_edge_filled_px": int(np.sum(dark_near_edge)),
    }
    return bg_layer, cumulative, meta
