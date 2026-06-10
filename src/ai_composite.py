import base64
import os
import time
import cv2
import numpy as np
import requests
from openai import OpenAI


API_KEY = "sk-e28ec115af2a4f2d95f475fb50e161e3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"
DASHSCOPE_GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime(path):
    ext = path.rsplit(".", 1)[-1].lower()
    return "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"


def _wait_task(task_id):
    for i in range(30):
        r = requests.get(f"{DASHSCOPE_TASK_URL}/{task_id}", headers={"Authorization": f"Bearer {API_KEY}"})
        d = r.json()
        status = d["output"]["task_status"]
        if status == "SUCCEEDED":
            return d
        elif status == "FAILED":
            return None
        time.sleep(5)
    return None


def analyze_features(portrait_path):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    img_b64 = _encode_image(portrait_path)
    mime = _get_mime(portrait_path)

    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a visual analysis assistant that can see and describe images."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": "Describe the specific facial features you see in this portrait photo. Then write an English prompt (under 50 words) for generating an exaggerated cartoon caricature of this person. Only output the English prompt."}
            ]}
        ],
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()


def generate_caricature(portrait_path, output_dir):
    feature_prompt = analyze_features(portrait_path)
    p_b64 = _encode_image(portrait_path)
    mime = _get_mime(portrait_path)

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-DashScope-Async": "enable"}
    data = {
        "model": "wan2.7-image-pro",
        "input": {"messages": [{"role": "user", "content": [
            {"type": "image", "image": f"data:{mime};base64,{p_b64}"},
            {"type": "text", "text": f"{feature_prompt} Preserve the exact identity of this person. Bold outlines, vibrant colors, cartoon caricature style."}
        ]}]},
        "parameters": {"size": "1024*1024", "n": 1, "prompt_extend": False}
    }

    r = requests.post(DASHSCOPE_GEN_URL, headers=headers, json=data)
    d = r.json()
    task_id = d.get("output", {}).get("task_id", "")
    if not task_id:
        return None, feature_prompt

    result = _wait_task(task_id)
    if result is None:
        return None, feature_prompt

    img_url = result["output"]["choices"][0]["message"]["content"][0]["image"]
    img_resp = requests.get(img_url, timeout=60)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_dir}/caricature_ai_{timestamp}.png"
    with open(out_path, "wb") as f:
        f.write(img_resp.content)
    return out_path, feature_prompt


def generate_composite_step1(portrait_path, target_path, output_dir):
    p_b64 = _encode_image(portrait_path)
    t_b64 = _encode_image(target_path)
    p_mime = _get_mime(portrait_path)
    t_mime = _get_mime(target_path)

    prompt = (
        "Take the person from photo 1 and transform them into the cute cat character from photo 2. "
        "CRITICAL: Keep the persons facial STRUCTURE (eye shape, nose shape, mouth shape, jawline, brow shape) "
        "but render the face using the cats TEXTURE and COLOR style from photo 2. "
        "The face should have orange fur texture, cartoon shading, and cute pastel coloring like the cat in photo 2. "
        "The persons deep-set eyes become deep-set orange fur-textured eyes. "
        "The persons hooked nose becomes an orange fur-textured hooked nose. "
        "Do NOT give big round white-circle cat eyes. Keep the persons eye shape. "
        "Keep the cats body, ears and posture exactly as photo 2. "
        "Exaggerated caricature style, bold outlines, vibrant colors."
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-DashScope-Async": "enable"}
    data = {
        "model": "wan2.7-image-pro",
        "input": {"messages": [{"role": "user", "content": [
            {"type": "image", "image": f"data:{p_mime};base64,{p_b64}"},
            {"type": "image", "image": f"data:{t_mime};base64,{t_b64}"},
            {"type": "text", "text": prompt}
        ]}]},
        "parameters": {"size": "1024*1024", "n": 1, "prompt_extend": False}
    }

    r = requests.post(DASHSCOPE_GEN_URL, headers=headers, json=data)
    task_id = r.json().get("output", {}).get("task_id", "")
    if not task_id:
        return None

    result = _wait_task(task_id)
    if result is None:
        return None

    img_url = result["output"]["choices"][0]["message"]["content"][0]["image"]
    img_resp = requests.get(img_url, timeout=60)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    step1_path = f"{output_dir}/composite_step1_{timestamp}.png"
    with open(step1_path, "wb") as f:
        f.write(img_resp.content)
    return step1_path


def create_eye_mask(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    left_cx, left_cy = int(w * 0.35), int(h * 0.28)
    right_cx, right_cy = int(w * 0.65), int(h * 0.28)
    radius = int(min(w, h) * 0.10)
    cv2.circle(mask, (left_cx, left_cy), radius, 255, -1)
    cv2.circle(mask, (right_cx, right_cy), radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_path = image_path.rsplit(".", 1)[0] + "_eyemask.png"
    cv2.imwrite(mask_path, mask)
    return mask_path


def fix_eyes(composite_path, portrait_path):
    img_b64 = _encode_image(composite_path)
    mask_path = create_eye_mask(composite_path)
    mask_b64 = _encode_image(mask_path)
    p_b64 = _encode_image(portrait_path)
    p_mime = _get_mime(portrait_path)

    prompt = (
        "Replace the big round white-circle cartoon eyes with deep-set eyes that match the persons eye shape. "
        "The eyes should have dark irises, narrow and angular shape like the person in the reference portrait. "
        "Render the eyes with the orange fur texture and cartoon coloring style of the surrounding cat face. "
        "Add thick downward-angled eyebrows above each eye with orange fur texture matching the persons brow shape. "
        "Keep everything else in the image exactly the same."
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-DashScope-Async": "enable"}
    data = {
        "model": "wan2.7-image-pro",
        "input": {"messages": [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
            {"type": "image", "image": f"data:image/png;base64,{mask_b64}"},
            {"type": "image", "image": f"data:{p_mime};base64,{p_b64}"},
            {"type": "text", "text": prompt}
        ]}]},
        "parameters": {"size": "1024*1024", "n": 1, "prompt_extend": False}
    }

    r = requests.post(DASHSCOPE_GEN_URL, headers=headers, json=data)
    task_id = r.json().get("output", {}).get("task_id", "")
    if not task_id:
        return None

    result = _wait_task(task_id)
    if result is None:
        return None

    img_url = result["output"]["choices"][0]["message"]["content"][0]["image"]
    img_resp = requests.get(img_url, timeout=60)
    output_dir = os.path.dirname(composite_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_dir}/composite_ai_{timestamp}.png"
    with open(out_path, "wb") as f:
        f.write(img_resp.content)
    return out_path


def generate_composite(portrait_path, target_path, output_dir):
    import os
    step1_path = generate_composite_step1(portrait_path, target_path, output_dir)
    if step1_path is None:
        return None, None
    result_path = fix_eyes(step1_path, portrait_path)
    return result_path, step1_path