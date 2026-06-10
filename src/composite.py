import cv2
import numpy as np
from src.face_landmark import FaceLandmarkDetector


def color_transfer(src, dst):
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
    dst_mean, dst_std = dst_lab.mean(axis=(0, 1)), dst_lab.std(axis=(0, 1))

    src_std = np.where(src_std < 1, 1, src_std)
    dst_std = np.where(dst_std < 1, 1, dst_std)

    result_lab = (dst_lab - dst_mean) * (src_std / dst_std) + src_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return result


def seamless_blend(src_face, target, center_point, mask):
    h, w = target.shape[:2]
    src_h, src_w = src_face.shape[:2]
    src_center = (src_w // 2, src_h // 2)

    try:
        result = cv2.seamlessClone(src_face, target, mask, center_point, cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        try:
            result = cv2.seamlessClone(src_face, target, mask, center_point, cv2.MIXED_CLONE)
            return result
        except cv2.error:
            mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
            blended = (src_face.astype(np.float32) * mask_3d +
                       target.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)
            return blended


def find_target_region(target):
    h, w = target.shape[:2]

    if target.shape[2] == 4:
        alpha = target[:, :, 3]
        mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, tw, th = cv2.boundingRect(largest)
            center = (x + tw // 2, y + th // 2)
            return center, (tw, th), mask
    else:
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, tw, th = cv2.boundingRect(largest)
            center = (x + tw // 2, y + th * 2 // 5)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(largest), 255)
            return center, (tw, th), mask

    center = (w // 2, h * 2 // 5)
    region_size = (min(w, h) // 2, min(w, h) // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, min(w, h) // 3, 255, -1)
    return center, region_size, mask


def create_feature_fused_composite(caricature, caricature_landmarks, target, target_center, target_size, target_mask):
    h, w = target.shape[:2]
    outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                       227, 127, 162, 21, 54, 103, 67, 109]
    face_pts = [caricature_landmarks[i] for i in outline_indices]
    hull = cv2.convexHull(np.array(face_pts, dtype=np.int32))
    fx, fy, ffw, ffh = cv2.boundingRect(hull)

    face_mask = np.zeros(caricature.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)

    face_crop = caricature[fy:fy + ffh, fx:fx + ffw]
    mask_crop = face_mask[fy:fy + ffh, fx:fx + ffw]

    tcx, tcy = target_center
    tw, th = target_size

    scale_x = tw / max(ffw, 1) * 0.9
    scale_y = th / max(ffh, 1) * 0.9
    scale = min(scale_x, scale_y)

    new_fw = int(ffw * scale)
    new_fh = int(ffh * scale)

    if new_fw <= 0 or new_fh <= 0:
        return target

    resized_face = cv2.resize(face_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)

    color_adapted_face = color_transfer(target, resized_face)

    padded_face = np.zeros((h, w, 3), dtype=np.uint8)
    padded_mask = np.zeros((h, w), dtype=np.uint8)

    paste_x = tcx - new_fw // 2
    paste_y = tcy - new_fh // 2

    sx1 = max(0, -paste_x)
    sy1 = max(0, -paste_y)
    sx2 = min(new_fw, new_fw - (paste_x + new_fw - w))
    sy2 = min(new_fh, new_fh - (paste_y + new_fh - h))

    dx1 = paste_x + sx1
    dy1 = paste_y + sy1
    dx2 = paste_x + sx2
    dy2 = paste_y + sy2

    if sx1 >= sx2 or sy1 >= sy2:
        return target

    padded_face[dy1:dy2, dx1:dx2] = color_adapted_face[sy1:sy2, sx1:sx2]
    padded_mask[dy1:dy2, dx1:dx2] = resized_mask[sy1:sy2, sx1:sx2]

    result = seamless_blend(padded_face, target, (tcx, tcy), padded_mask)

    alpha = 0.3
    mask_float = padded_mask.astype(np.float32) / 255.0
    mask_3d = mask_float[:, :, np.newaxis]
    blended = (result.astype(np.float32) * mask_3d * alpha +
               target.astype(np.float32) * (1.0 - mask_3d * alpha)).astype(np.uint8)

    edge_overlay = (result.astype(np.float32) * mask_3d * 0.15 +
                    blended.astype(np.float32) * (1.0 - mask_3d * 0.15)).astype(np.uint8)

    return edge_overlay


def create_composite(caricature_path_or_image, portrait_landmarks_path, target_path):
    detector = FaceLandmarkDetector()

    if isinstance(caricature_path_or_image, str):
        caricature = cv2.imread(caricature_path_or_image)
    else:
        caricature = caricature_path_or_image

    if isinstance(portrait_landmarks_path, list):
        landmarks = portrait_landmarks_path
    else:
        points, _ = detector.detect(portrait_landmarks_path)
        if points is None:
            detector.release()
            return None
        landmarks = points

    target_img = cv2.imread(target_path)
    if target_img is None:
        from PIL import Image
        pil_img = Image.open(target_path).convert("RGBA")
        target_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

    target_center, target_size, target_mask = find_target_region(target_img)

    result = create_feature_fused_composite(
        caricature, landmarks, target_img,
        target_center, target_size, target_mask
    )

    detector.release()
    return result