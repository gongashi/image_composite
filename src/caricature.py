import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


def warp_image_local(image, src_points, dst_points, radius=50):
    h, w = image.shape[:2]
    result = image.copy()

    src_arr = np.array(src_points, dtype=np.float64)
    dst_arr = np.array(dst_points, dtype=np.float64)

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    displacements = dst_arr - src_arr

    try:
        interpolator = RBFInterpolator(src_arr, displacements, kernel="thin_plate_spline", smoothing=0.1)
        warp_field = interpolator(grid_points)
    except Exception:
        for i, (sp, dp) in enumerate(zip(src_arr, dst_arr)):
            dx, dy = dp - sp
            for gy in range(max(0, int(sp[1] - radius)), min(h, int(sp[1] + radius))):
                for gx in range(max(0, int(sp[0] - radius)), min(w, int(sp[0] + radius))):
                    dist = np.sqrt((gx - sp[0]) ** 2 + (gy - sp[1]) ** 2)
                    if dist < radius:
                        weight = 1.0 - (dist / radius) ** 2
                        new_x = int(gx + dx * weight)
                        new_y = int(gy + dy * weight)
                        if 0 <= new_x < w and 0 <= new_y < h:
                            result[gy, gx] = image[new_y, new_x]
        return result

    warp_x = warp_field[:, 0].reshape(h, w)
    warp_y = warp_field[:, 1].reshape(h, w)

    map_x = (grid_x + warp_x).astype(np.float32)
    map_y = (grid_y + warp_y).astype(np.float32)

    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return result


def exaggerate_features(image, features, ratios, exaggeration_factor=1.8):
    h, w = image.shape[:2]
    all_points = []
    dst_points = []

    center = features["center"][0]

    face_height = ratios["face_height"]
    face_width = ratios["face_width"]

    for feature_name, feature_pts in features.items():
        if feature_name in ("center", "forehead", "chin", "nose_tip"):
            continue

        factor = exaggeration_factor

        if "nose" in feature_name:
            nose_width_ratio = ratios.get("nose_width_ratio", 0.2)
            if nose_width_ratio > 0.25:
                factor = exaggeration_factor * 1.3
            nose_length_ratio = ratios.get("nose_length_ratio", 0.4)
            if nose_length_ratio > 0.45:
                factor = exaggeration_factor * 1.2

        if "eye" in feature_name and "brow" not in feature_name:
            eye_spacing_ratio = ratios.get("eye_spacing_ratio", 0.4)
            if eye_spacing_ratio > 0.45:
                factor = exaggeration_factor * 0.8
            elif eye_spacing_ratio < 0.35:
                factor = exaggeration_factor * 1.4

        if "mouth" in feature_name:
            mouth_width_ratio = ratios.get("mouth_width_ratio", 0.35)
            if mouth_width_ratio > 0.4:
                factor = exaggeration_factor * 1.3
            elif mouth_width_ratio < 0.25:
                factor = exaggeration_factor * 0.7

        for pt in feature_pts:
            all_points.append(pt)
            dx = (pt[0] - center[0]) * factor
            dy = (pt[1] - center[1]) * factor
            new_x = center[0] + dx
            new_y = center[1] + dy
            new_x = max(0, min(w - 1, new_x))
            new_y = max(0, min(h - 1, new_y))
            dst_points.append((new_x, new_y))

    for key in ["center", "nose_tip"]:
        if key in features:
            pt = features[key]
            all_points.append(pt[0])
            dst_points.append(pt[0])

    warped = warp_image_local(image, all_points, dst_points)
    return warped


def apply_cartoon_style(image, saturation_boost=1.5, edge_thickness=3):
    img = image.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges = cv2.dilate(edges, np.ones((edge_thickness, edge_thickness), np.uint8))
    edges_inv = cv2.bitwise_not(edges)

    img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    result = cv2.bitwise_and(img_color, img_color, mask=edges_inv)
    return result


def create_caricature(image_path, exaggeration_factor=1.8, cartoon_style=True):
    from src.face_landmark import FaceLandmarkDetector

    detector = FaceLandmarkDetector()
    points, image = detector.detect(image_path)
    if points is None:
        detector.release()
        return None

    features = detector.get_feature_regions(points)
    ratios = detector.compute_feature_ratios(features, image.shape)

    exaggerated = exaggerate_features(image, features, ratios, exaggeration_factor)

    if cartoon_style:
        result = apply_cartoon_style(exaggerated)
    else:
        result = exaggerated

    detector.release()
    return result, features, ratios