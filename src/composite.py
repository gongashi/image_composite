import cv2
import numpy as np
from src.face_landmark import FaceLandmarkDetector


def partial_color_blend(src_face, target_region, blend_ratio=0.3):
    src_lab = cv2.cvtColor(src_face, cv2.COLOR_BGR2LAB).astype(np.float64)
    tgt_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float64)
    tgt_mean = tgt_lab.mean(axis=(0, 1))
    result = src_lab * (1.0 - blend_ratio) + tgt_mean * blend_ratio
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def extract_face_edges(caricature, landmarks):
    outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                       227, 127, 162, 21, 54, 103, 67, 109]
    face_pts = [landmarks[i] for i in outline_indices]
    hull = cv2.convexHull(np.array(face_pts, dtype=np.int32))
    mask = np.zeros(caricature.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    edges = cv2.Canny(caricature, 80, 200)
    edges_on_face = cv2.bitwise_and(edges, edges, mask=mask)
    return edges_on_face, hull


def find_target_center_and_region(target):
    h, w = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, tw, th = cv2.boundingRect(largest)
        cx = x + tw // 2
        cy = y + int(th * 0.4)
        return (cx, cy), (tw, th)
    return (w // 2, int(h * 0.4)), (min(w, h) // 2, min(w, h) // 2)


def create_composite(caricature_path_or_image, portrait_landmarks_path, target_path):
    detector = FaceLandmarkDetector()

    if isinstance(caricature_path_or_image, str):
        caricature = cv2.imread(caricature_path_or_image)
    else:
        caricature = caricature_path_or_image.copy()

    if isinstance(portrait_landmarks_path, list):
        landmarks = portrait_landmarks_path
    else:
        points, _ = detector.detect(portrait_landmarks_path)
        if points is None:
            detector.release()
            return None
        landmarks = points

    target = cv2.imread(target_path)
    if target is None:
        from PIL import Image
        pil_img = Image.open(target_path).convert("RGBA")
        arr = np.array(pil_img)
        target = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

    outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                       227, 127, 162, 21, 54, 103, 67, 109]
    face_pts = [landmarks[i] for i in outline_indices]
    hull = cv2.convexHull(np.array(face_pts, dtype=np.int32))
    fx, fy, ffw, ffh = cv2.boundingRect(hull)

    face_crop = caricature[fy:fy + ffh, fx:fx + ffw].copy()
    face_mask = np.zeros(caricature.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)
    face_mask_crop = face_mask[fy:fy + ffh, fx:fx + ffw]
    face_mask_crop = cv2.GaussianBlur(face_mask_crop, (21, 21), 0)

    face_edges, _ = extract_face_edges(caricature, landmarks)
    face_edges_crop = face_edges[fy:fy + ffh, fx:fx + ffw]

    target_center, target_size = find_target_center_and_region(target)
    tcx, tcy = target_center
    tw, th = target_size

    scale = min(tw / max(ffw, 1), th / max(ffh, 1)) * 0.8
    new_fw = max(1, int(ffw * scale))
    new_fh = max(1, int(ffh * scale))

    resized_face = cv2.resize(face_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(face_mask_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
    resized_edges = cv2.resize(face_edges_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)

    target_region = target[max(0,tcy-new_fh//2):min(target.shape[0],tcy+new_fh//2),
                           max(0,tcx-new_fw//2):min(target.shape[1],tcx+new_fw//2)]
    color_adapted = partial_color_blend(resized_face, target_region, blend_ratio=0.25)

    th_img, tw_img = target.shape[:2]
    paste_x = tcx - new_fw // 2
    paste_y = tcy - new_fh // 2

    result = target.copy()

    sx1 = max(0, -paste_x)
    sy1 = max(0, -paste_y)
    dx1 = max(0, paste_x)
    dy1 = max(0, paste_y)
    sx2 = new_fw - max(0, paste_x + new_fw - tw_img)
    sy2 = new_fh - max(0, paste_y + new_fh - th_img)
    dx2 = tw_img - max(0, tw_img - (paste_x + new_fw))
    dy2 = th_img - max(0, th_img - (paste_y + new_fh))

    if sx1 >= sx2 or sy1 >= sy2:
        detector.release()
        return result

    face_roi = color_adapted[sy1:sy2, sx1:sx2]
    mask_roi = resized_mask[sy1:sy2, sx1:sx2]
    edges_roi = resized_edges[sy1:sy2, sx1:sx2]

    mask_float = mask_roi.astype(np.float32) / 255.0
    mask_3d = mask_float[:, :, np.newaxis]

    alpha_face = 0.6
    alpha_edges = 0.7

    blended_face = (face_roi.astype(np.float32) * mask_3d * alpha_face +
                    result[dy1:dy2, dx1:dx2].astype(np.float32) * (1.0 - mask_3d * alpha_face)).astype(np.uint8)

    edges_colored = np.zeros_like(blended_face)
    edges_colored[edges_roi > 0] = [30, 30, 30]
    edges_mask = (edges_roi > 0).astype(np.float32)[:, :, np.newaxis]

    blended = (blended_face.astype(np.float32) * (1.0 - edges_mask * alpha_edges) +
               edges_colored.astype(np.float32) * edges_mask * alpha_edges).astype(np.uint8)

    result[dy1:dy2, dx1:dx2] = blended

    detector.release()
    return result