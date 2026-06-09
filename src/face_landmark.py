import cv2
import numpy as np
import os
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import Image as MpImage, ImageFormat

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")


class FaceLandmarkDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None, image
        landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]
        points = []
        for lm in landmarks:
            points.append((int(lm.x * w), int(lm.y * h)))
        return points, image

    def get_feature_regions(self, points):
        features = {}

        left_eye = points[33:42] + [points[133], points[157], points[158], points[159]]
        right_eye = points[263:272] + [points[362], points[386], points[387], points[388]]
        features["left_eye"] = left_eye
        features["right_eye"] = right_eye

        nose_tip = [points[1]]
        nose_bridge = [points[6], points[197], points[195], points[5]]
        nose_left = points[48:55] + [points[60], points[61], points[276]]
        nose_right = points[278:285] + [points[290], points[291], points[4]]
        features["nose_tip"] = nose_tip
        features["nose_bridge"] = nose_bridge
        features["nose_left"] = nose_left
        features["nose_right"] = nose_right
        features["nose_full"] = nose_bridge + nose_left + nose_right + nose_tip

        mouth_outer = [points[61], points[146], points[91], points[181], points[84],
                       points[17], points[314], points[405], points[321], points[375],
                       points[291], points[308], points[78], points[191], points[80],
                       points[13], points[312], points[311], points[310], points[415]]
        mouth_inner = [points[78], points[191], points[80], points[13],
                       points[312], points[311], points[310], points[415]]
        features["mouth_outer"] = mouth_outer
        features["mouth_inner"] = mouth_inner

        left_eyebrow = [points[70], points[63], points[105], points[66], points[107],
                        points[55], points[65], points[52], points[53]]
        right_eyebrow = [points[300], points[293], points[334], points[296], points[337],
                         points[285], points[295], points[282], points[283]]
        features["left_eyebrow"] = left_eyebrow
        features["right_eyebrow"] = right_eyebrow

        face_outline = [points[10], points[338], points[297], points[332], points[284],
                        points[251], points[389], points[356], points[454], points[323],
                        points[361], points[288], points[397], points[365], points[379],
                        points[378], points[400], points[377], points[152], points[148],
                        points[176], points[149], points[150], points[136], points[172],
                        points[58], points[132], points[177], points[215], points[137],
                        points[227], points[127], points[162], points[21], points[54],
                        points[103], points[67], points[109]]
        features["face_outline"] = face_outline

        features["center"] = [points[5]]
        features["forehead"] = [points[10]]
        features["chin"] = [points[152]]

        return features

    def compute_feature_ratios(self, features, image_shape):
        h, w = image_shape[:2]
        ratios = {}

        center = features["center"][0]
        forehead = features["forehead"][0]
        chin = features["chin"][0]
        face_height = abs(chin[1] - forehead[1])
        face_width = max(abs(p[0] - center[0]) for p in features["face_outline"])

        ratios["face_height"] = face_height
        ratios["face_width"] = face_width
        ratios["face_hw_ratio"] = face_height / max(face_width, 1)

        nose_tip = features["nose_tip"][0]
        nose_width = max(abs(p[0] - nose_tip[0]) for p in features["nose_left"] + features["nose_right"])
        nose_length = abs(nose_tip[1] - features["nose_bridge"][0][1])
        ratios["nose_width_ratio"] = nose_width / max(face_width, 1)
        ratios["nose_length_ratio"] = nose_length / max(face_height, 1)
        ratios["nose_width"] = nose_width
        ratios["nose_length"] = nose_length

        left_eye_pts = features["left_eye"]
        right_eye_pts = features["right_eye"]
        left_eye_w = max(abs(p[0] - left_eye_pts[0][0]) for p in left_eye_pts)
        left_eye_h = max(abs(p[1] - left_eye_pts[0][1]) for p in left_eye_pts)
        right_eye_w = max(abs(p[0] - right_eye_pts[0][0]) for p in right_eye_pts)
        right_eye_h = max(abs(p[1] - right_eye_pts[0][1]) for p in right_eye_pts)
        eye_spacing = abs(right_eye_pts[0][0] - left_eye_pts[0][0])
        ratios["left_eye_size"] = (left_eye_w, left_eye_h)
        ratios["right_eye_size"] = (right_eye_w, right_eye_h)
        ratios["eye_spacing"] = eye_spacing
        ratios["eye_spacing_ratio"] = eye_spacing / max(face_width, 1)

        mouth_pts = features["mouth_outer"]
        mouth_w = max(abs(p[0] - mouth_pts[0][0]) for p in mouth_pts)
        mouth_h = max(abs(p[1] - mouth_pts[0][1]) for p in mouth_pts)
        ratios["mouth_width"] = mouth_w
        ratios["mouth_height"] = mouth_h
        ratios["mouth_width_ratio"] = mouth_w / max(face_width, 1)

        return ratios

    def release(self):
        self.landmarker.close()