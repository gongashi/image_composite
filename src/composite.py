import cv2
import numpy as np
from src.face_landmark import FaceLandmarkDetector


def find_target_center_and_size(target):
    h, w = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    bg_val = gray[0, 0]
    fg_mask = cv2.threshold(gray, int(bg_val) + 20 if bg_val < 200 else 30,
                            255, cv2.THRESH_BINARY_INV if bg_val > 200 else cv2.THRESH_BINARY)[1]
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, tw, th = cv2.boundingRect(largest)
        cx = x + tw // 2
        cy = y + int(th * 0.4)
        return (cx, cy), (tw, th)

    try:
        from PIL import Image
        pil = Image.open(target_path)
        if pil.mode == 'RGBA':
            alpha = np.array(pil)[:, :, 3]
            a_contours = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if a_contours:
                largest = max(a_contours, key=cv2.contourArea)
                x, y, tw, th = cv2.boundingRect(largest)
                return (x + tw // 2, y + int(th * 0.4)), (tw, th)
    except Exception:
        pass

    return (w // 2, int(h * 0.4)), (int(w * 0.6), int(h * 0.6))


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

    target_center, target_size = find_target_center_and_size(target)
    tcx, tcy = target_center
    tw, th = target_size

    scale = min(tw / max(ffw, 1), th / max(ffh, 1)) * 0.85
    new_fw = max(1, int(ffw * scale))
    new_fh = max(1, int(ffh * scale))

    resized_face = cv2.resize(face_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(face_mask_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)

    th_img, tw_img = target.shape[:2]
    result = target.copy()

    paste_x = tcx - new_fw // 2
    paste_y = tcy - new_fh // 2

    sx1 = max(0, -paste_x)
    sy1 = max(0, -paste_y)
    sx2 = new_fw - max(0, paste_x + new_fw - tw_img)
    sy2 = new_fh - max(0, paste_y + new_fh - th_img)
    dx1 = max(0, paste_x)
    dy1 = max(0, paste_y)
    dx2 = tw_img - max(0, tw_img - (paste_x + new_fw))
    dy2 = th_img - max(0, th_img - (paste_y + new_fh))

    if sx1 >= sx2 or sy1 >= sy2 or dx1 >= dx2 or dy1 >= dy2:
        detector.release()
        return target

    roi = result[dy1:dy2, dx1:dx2]
    face_roi = resized_face[sy1:sy2, sx1:sx2]
    mask_roi = resized_mask[sy1:sy2, sx1:sx2]

    mask_float = mask_roi.astype(np.float32) / 255.0
    mask_3d = mask_float[:, :, np.newaxis]

    alpha = 0.7
    blended = (face_roi.astype(np.float32) * mask_3d * alpha +
               roi.astype(np.float32) * (1.0 - mask_3d * alpha)).astype(np.uint8)

    result[dy1:dy2, dx1:dx2] = blended

    detector.release()
    return result