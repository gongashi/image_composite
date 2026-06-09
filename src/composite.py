import cv2
import numpy as np
from src.face_landmark import FaceLandmarkDetector


def extract_face_mask(image, landmarks, feather_amount=20):
    h, w = image.shape[:2]
    outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                       227, 127, 162, 21, 54, 103, 67, 109]
    points = [landmarks[i] for i in outline_indices if i < len(landmarks)]

    hull = cv2.convexHull(np.array(points, dtype=np.int32))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    mask_f = mask.astype(np.float32) / 255.0
    mask_blurred = cv2.GaussianBlur(mask_f, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
    mask_blurred = np.clip(mask_blurred, 0, 1)

    return mask_blurred


def extract_face_region(image, landmarks):
    outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                       227, 127, 162, 21, 54, 103, 67, 109]
    points = [landmarks[i] for i in outline_indices if i < len(landmarks)]

    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = extract_face_mask(image, landmarks, feather_amount=30)

    face_region = (image.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)

    x, y, fw, fh = cv2.boundingRect(hull)
    face_crop = face_region[y:y + fh, x:x + fw]
    mask_crop = mask[y:y + fh, x:x + fw]

    return face_crop, mask_crop, (x, y, fw, fh)


def detect_target_face_or_region(target_image_path):
    detector = FaceLandmarkDetector()
    points, image = detector.detect(target_image_path)
    detector.release()

    if points is not None:
        outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                           361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                           176, 149, 150, 136, 172, 58, 132, 177, 215, 137,
                           227, 127, 162, 21, 54, 103, 67, 109]
        face_pts = [points[i] for i in outline_indices if i < len(points)]
        hull = cv2.convexHull(np.array(face_pts, dtype=np.int32))
        x, y, fw, fh = cv2.boundingRect(hull)
        return image, "face", (x, y, fw, fh)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, fw, fh = cv2.boundingRect(largest)
            center_x = x + fw // 3
            center_y = y + fh // 3
            region_w = fw // 2
            region_h = fh // 2
            return image, "object", (center_x - region_w // 2, center_y - region_h // 2, region_w, region_h)
        else:
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 3
            rw, rh = w // 2, h // 2
            return image, "center", (cx - rw // 2, cy - rh // 2, rw, rh)


def blend_face_on_target(caricature_image, caricature_landmarks, target_image, target_info):
    target_img = target_image.copy()
    target_type, target_bbox = target_info

    face_crop, mask_crop, face_bbox = extract_face_region(caricature_image, caricature_landmarks)
    fx, fy, fw, fh = face_bbox

    tx, ty, tw, th = target_bbox

    scale_x = tw / max(fw, 1) * 0.8
    scale_y = th / max(fh, 1) * 0.8
    scale = min(scale_x, scale_y)

    new_fw = int(fw * scale)
    new_fh = int(fh * scale)

    if new_fw <= 0 or new_fh <= 0:
        return target_img

    resized_face = cv2.resize(face_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_crop, (new_fw, new_fh), interpolation=cv2.INTER_AREA)

    center_x = tx + tw // 2
    center_y = ty + th // 2

    paste_x = center_x - new_fw // 2
    paste_y = center_y - new_fh // 2

    target_h, target_w = target_img.shape[:2]

    src_x_start = 0
    src_y_start = 0
    src_x_end = new_fw
    src_y_end = new_fh

    dst_x_start = paste_x
    dst_y_start = paste_y
    dst_x_end = paste_x + new_fw
    dst_y_end = paste_y + new_fh

    if dst_x_start < 0:
        src_x_start = -dst_x_start
        dst_x_start = 0
    if dst_y_start < 0:
        src_y_start = -dst_y_start
        dst_y_start = 0
    if dst_x_end > target_w:
        src_x_end -= (dst_x_end - target_w)
        dst_x_end = target_w
    if dst_y_end > target_h:
        src_y_end -= (dst_y_end - target_h)
        dst_y_end = target_h

    if src_x_start >= src_x_end or src_y_start >= src_y_end:
        return target_img

    roi = target_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
    face_roi = resized_face[src_y_start:src_y_end, src_x_start:src_x_end]
    mask_roi = resized_mask[src_y_start:src_y_end, src_x_start:src_x_end]

    mask_3d = mask_roi[:, :, np.newaxis]
    blended = (face_roi.astype(np.float32) * mask_3d +
               roi.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)

    target_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended

    return target_img


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

    target_img, target_type, target_bbox = detect_target_face_or_region(target_path)

    result = blend_face_on_target(caricature, landmarks, target_img, (target_type, target_bbox))

    detector.release()
    return result