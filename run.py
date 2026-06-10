import cv2
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face_landmark import FaceLandmarkDetector
from src.caricature import create_caricature
from src.composite import create_composite
from src.ai_composite import generate_caricature, generate_composite, analyze_features, generate_composite_step1, create_eye_mask, fix_eyes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PORTRAITS_DIR = os.path.join(BASE_DIR, "input", "portraits")
EMOJIS_DIR = os.path.join(BASE_DIR, "input", "emojis")


def run_local(portrait_path, target_path, exaggeration=1.8):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[1/3] 检测人像面部特征...")
    detector = FaceLandmarkDetector()
    landmarks, image = detector.detect(portrait_path)
    if landmarks is None:
        print("ERROR: 未检测到人脸！")
        detector.release()
        return None, None

    features = detector.get_feature_regions(landmarks)
    ratios = detector.compute_feature_ratios(features, image.shape)
    print(f"  面部宽高比: {ratios['face_hw_ratio']:.2f}")
    print(f"  鼻宽/面宽: {ratios['nose_width_ratio']:.2f}")
    print(f"  眼间距/面宽: {ratios['eye_spacing_ratio']:.2f}")
    print(f"  嘴宽/面宽: {ratios['mouth_width_ratio']:.2f}")
    detector.release()

    print(f"[2/3] 生成漫画化人像 (夸张系数={exaggeration})...")
    result_caricature, _, _ = create_caricature(portrait_path, exaggeration_factor=exaggeration)
    if result_caricature is None:
        print("ERROR: 漫画化失败！")
        return None, None

    caricature_path = os.path.join(OUTPUT_DIR, f"caricature_{timestamp}.png")
    cv2.imwrite(caricature_path, result_caricature)
    print(f"  已保存: {caricature_path}")

    print(f"[3/3] 合成漫画人像 + 目标图片...")
    result_composite = create_composite(result_caricature, landmarks, target_path)
    if result_composite is None:
        print("ERROR: 合成失败！")
        return caricature_path, None

    composite_path = os.path.join(OUTPUT_DIR, f"composite_{timestamp}.png")
    cv2.imwrite(composite_path, result_composite)
    print(f"  已保存: {composite_path}")

    print(f"\n=== 完成 (本地模式) ===")
    print(f"漫画化人像: {caricature_path}")
    print(f"合成结果: {composite_path}")
    return caricature_path, composite_path


def run_ai(portrait_path, target_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/4] AI分析人像特征 + 生成漫画化人像...")
    caricature_path, car_prompt = generate_caricature(portrait_path, OUTPUT_DIR)
    if caricature_path is None:
        print("ERROR: AI漫画化生成失败！")
        print(f"  Prompt: {car_prompt}")
        return None, None
    print(f"  Prompt: {car_prompt}")
    print(f"  已保存: {caricature_path}")

    print(f"[2/4] AI生成初步合成图(人像+表情包)...")
    step1_path = generate_composite_step1(portrait_path, target_path, OUTPUT_DIR)
    if step1_path is None:
        print("ERROR: 初步合成失败！")
        return caricature_path, None
    print(f"  已保存: {step1_path}")

    print(f"[3/4] 创建眼部修复mask...")
    mask_path = create_eye_mask(step1_path)
    print(f"  Mask: {mask_path}")

    print(f"[4/4] AI局部重绘修复眼睛区域...")
    from src.ai_composite import fix_eyes
    composite_path = fix_eyes(step1_path, portrait_path)
    if composite_path is None:
        print("WARNING: 眼部修复失败，使用初步合成结果")
        composite_path = step1_path
    print(f"  已保存: {composite_path}")

    print(f"\n=== 完成 (AI模式) ===")
    print(f"漫画化人像: {caricature_path}")
    print(f"合成结果: {composite_path}")
    return caricature_path, composite_path


def run_hybrid(portrait_path, target_path, exaggeration=1.8):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[1/5] 本地检测人像特征...")
    detector = FaceLandmarkDetector()
    landmarks, image = detector.detect(portrait_path)
    if landmarks is None:
        print("ERROR: 未检测到人脸！")
        detector.release()
        return None, None

    features = detector.get_feature_regions(landmarks)
    ratios = detector.compute_feature_ratios(features, image.shape)
    print(f"  面部宽高比: {ratios['face_hw_ratio']:.2f}")
    print(f"  鼻宽/面宽: {ratios['nose_width_ratio']:.2f}")
    detector.release()

    print(f"[2/5] 本地生成漫画化人像...")
    result_caricature, _, _ = create_caricature(portrait_path, exaggeration_factor=exaggeration)
    if result_caricature is None:
        print("ERROR: 漫画化失败！")
        return None, None

    caricature_path = os.path.join(OUTPUT_DIR, f"caricature_hybrid_{timestamp}.png")
    cv2.imwrite(caricature_path, result_caricature)
    print(f"  已保存: {caricature_path}")

    print(f"[3/5] AI生成初步合成图(人像结构+表情包纹理)...")
    step1_path = generate_composite_step1(portrait_path, target_path, OUTPUT_DIR)
    if step1_path is None:
        print("ERROR: 初步合成失败！")
        return caricature_path, None
    print(f"  已保存: {step1_path}")

    print(f"[4/5] 创建眼部修复mask...")
    mask_path = create_eye_mask(step1_path)
    print(f"  Mask: {mask_path}")

    print(f"[5/5] AI局部重绘修复眼睛(人像形状+表情包纹理)...")
    composite_path = fix_eyes(step1_path, portrait_path)
    if composite_path is None:
        print("WARNING: 眼部修复失败，使用初步合成结果")
        composite_path = step1_path
    print(f"  已保存: {composite_path}")

    print(f"\n=== 完成 (混合模式) ===")
    print(f"漫画化人像(本地): {caricature_path}")
    print(f"合成结果(AI): {composite_path}")
    return caricature_path, composite_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python run.py <人像照片> <目标图片> [模式] [夸张系数]")
        print("模式: local(本地), ai(AI), hybrid(混合-推荐)")
        print("示例: python run.py input/portraits/photo.jpg input/emojis/cat.png ai")
        sys.exit(1)

    portrait = sys.argv[1]
    if not os.path.exists(portrait):
        portrait = os.path.join(PORTRAITS_DIR, portrait)
    if not os.path.exists(portrait):
        portrait = os.path.join(BASE_DIR, portrait)
    target = sys.argv[2]
    if not os.path.exists(target):
        target = os.path.join(EMOJIS_DIR, target)
    if not os.path.exists(target):
        target = os.path.join(BASE_DIR, target)

    mode = sys.argv[3] if len(sys.argv) > 3 else "ai"
    exag = float(sys.argv[4]) if len(sys.argv) > 4 else 1.8

    if mode == "local":
        run_local(portrait, target, exag)
    elif mode == "ai":
        run_ai(portrait, target)
    elif mode == "hybrid":
        run_hybrid(portrait, target, exag)
    else:
        print(f"未知模式: {mode}，可选: local, ai, hybrid")
        sys.exit(1)