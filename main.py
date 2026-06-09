import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_landmark import FaceLandmarkDetector
from src.caricature import create_caricature
from src.composite import create_composite


def process(portrait_path, target_path, exaggeration=1.8, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] 正在检测人像面部特征...")
    detector = FaceLandmarkDetector()
    landmarks, image = detector.detect(portrait_path)
    if landmarks is None:
        print("ERROR: 未能检测到人像面部！请确保照片中有清晰的人脸。")
        detector.release()
        return None

    features = detector.get_feature_regions(landmarks)
    ratios = detector.compute_feature_ratios(features, image.shape)

    print(f"  面部宽高比: {ratios['face_hw_ratio']:.2f}")
    print(f"  鼻子宽度比: {ratios['nose_width_ratio']:.2f}")
    print(f"  眼间距比: {ratios['eye_spacing_ratio']:.2f}")
    print(f"  嘴宽比: {ratios['mouth_width_ratio']:.2f}")

    detector.release()

    print(f"[2/3] 正在生成漫画化人像...")
    result_caricature, features_out, ratios_out = create_caricature(
        portrait_path, exaggeration_factor=exaggeration, cartoon_style=True
    )
    if result_caricature is None:
        print("ERROR: 漫画化处理失败！")
        return None

    caricature_path = os.path.join(output_dir, "caricature.png")
    cv2.imwrite(caricature_path, result_caricature)
    print(f"  漫画化结果已保存: {caricature_path}")

    print(f"[3/3] 正在合成人像+目标图片...")
    result_composite = create_composite(result_caricature, landmarks, target_path)
    if result_composite is None:
        print("ERROR: 合成处理失败！")
        return None

    composite_path = os.path.join(output_dir, "composite.png")
    cv2.imwrite(composite_path, result_composite)
    print(f"  合成结果已保存: {composite_path}")

    print("\n=== 处理完成 ===")
    print(f"输出1 - 漫画化人像: {caricature_path}")
    print(f"输出2 - 人像+目标合成: {composite_path}")

    return caricature_path, composite_path


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PORTRAITS_DIR = os.path.join(BASE_DIR, "input", "portraits")
    EMOJIS_DIR = os.path.join(BASE_DIR, "input", "emojis")

    if len(sys.argv) < 3:
        print("用法: python main.py <人像照片> <目标图片(表情/宠物)> [夸张系数]")
        print("人像照片放入 input/portraits/, 表情图片放入 input/emojis/")
        print("示例: python main.py input/portraits/photo.jpg input/emojis/cat.png")
        sys.exit(1)

    portrait_path = sys.argv[1]
    if not os.path.isabs(portrait_path):
        portrait_path = os.path.join(PORTRAITS_DIR, portrait_path)
    target_path = sys.argv[2]
    if not os.path.isabs(target_path):
        target_path = os.path.join(EMOJIS_DIR, target_path)
    exaggeration = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8

    output_dir = os.path.join(BASE_DIR, "output")

    result = process(portrait_path, target_path, exaggeration, output_dir)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()