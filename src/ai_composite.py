import base64
import time
import json
import requests
from openai import OpenAI


API_KEY = "sk-e28ec115af2a4f2d95f475fb50e161e3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"
DASHSCOPE_GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_features(portrait_path):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    img_b64 = _encode_image(portrait_path)
    ext = portrait_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "仔细观察这张人像照片，描述面部最突出的特征（脸型、眼睛大小和间距、"
                        "鼻子形状和大小、嘴巴宽度和形状、眉毛特点）。"
                        "然后给出一个简短的英文prompt，用于生成一张漫画风格图片，"
                        "将这些特征夸张化放大。prompt格式示例："
                        "'A cartoon caricature portrait of a person with exaggerated [features], comic book style'"
                        "只输出英文prompt，不要其他内容。"
                    )}
                ]
            }
        ],
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()


def analyze_target(target_path):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    img_b64 = _encode_image(target_path)
    ext = target_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "描述这张图中的主体（表情包或宠物），包括它的外观、颜色、姿态等关键特征。"
                        "然后给出一个简短的英文prompt来描述它。"
                        "只输出英文prompt，不要其他内容。"
                    )}
                ]
            }
        ],
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()


def generate_composite_prompt(portrait_path, target_path):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    portrait_b64 = _encode_image(portrait_path)
    target_b64 = _encode_image(target_path)
    p_ext = portrait_path.rsplit(".", 1)[-1].lower()
    t_ext = target_path.rsplit(".", 1)[-1].lower()
    p_mime = "image/jpeg" if p_ext in ("jpg", "jpeg") else "image/png"
    t_mime = "image/jpeg" if t_ext in ("jpg", "jpeg") else "image/png"

    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的图像生成prompt工程师，擅长描述人像特征融合到表情包/宠物上的创意合成。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{p_mime};base64,{portrait_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{t_mime};base64,{target_b64}"}},
                    {"type": "text", "text": (
                        "第一张是人像照片，第二张是表情包/宠物图片。\n"
                        "请仔细分析人像的面部特征（脸型、眼睛、鼻子、嘴巴、眉毛等突出特点），"
                        "然后生成一个详细的英文prompt，将人像的夸张漫画化面部特征融合到表情包/宠物上。\n"
                        "要求：\n"
                        "- 保留表情包/宠物的基本外观和风格\n"
                        "- 将人像最突出的面部特征（如尖下巴、凤凰眼、高鼻梁、宽嘴巴等）融合到表情包/宠物面部\n"
                        "- 漫画夸张风格\n"
                        "- 色彩鲜明，细节丰富\n\n"
                        "只输出英文prompt，不要其他内容。prompt控制在100词以内。"
                    )}
                ]
            }
        ],
        max_tokens=500
    )
    return resp.choices[0].message.content.strip()


def generate_caricature_with_wanx(portrait_path, output_dir, exaggeration_hint=None):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    img_b64 = _encode_image(portrait_path)
    ext = portrait_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "观察这张人像照片，描述面部最突出的可以夸张化的特征，"
                        "然后生成一个英文prompt用于创建漫画化人像。\n"
                        "要求：夸张化放大最突出的面部特征（如大鼻子更大、尖下巴更尖、凤凰眼更上挑等），"
                        "漫画风格，色彩鲜明，线条清晰。\n"
                        "只输出英文prompt，不要其他内容。"
                    )}
                ]
            }
        ],
        max_tokens=300
    )
    prompt = resp.choices[0].message.content.strip()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    data = {
        "model": "wanx-v1",
        "input": {"prompt": prompt},
        "parameters": {"size": "1024*1024", "n": 1}
    }
    resp = requests.post(DASHSCOPE_GEN_URL, headers=headers, json=data)
    task_id = resp.json()["output"]["task_id"]

    for _ in range(30):
        r = requests.get(f"{DASHSCOPE_TASK_URL}/{task_id}", headers={"Authorization": f"Bearer {API_KEY}"})
        d = r.json()
        status = d["output"]["task_status"]
        if status == "SUCCEEDED":
            img_url = d["output"]["results"][0]["url"]
            img_resp = requests.get(img_url, timeout=30)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"{output_dir}/caricature_ai_{timestamp}.png"
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            return out_path, prompt
        elif status == "FAILED":
            return None, prompt
        time.sleep(3)
    return None, prompt


def generate_composite_with_wanx(portrait_path, target_path, output_dir):
    prompt = generate_composite_prompt(portrait_path, target_path)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    data = {
        "model": "wanx-v1",
        "input": {"prompt": prompt},
        "parameters": {"size": "1024*1024", "n": 1}
    }
    resp = requests.post(DASHSCOPE_GEN_URL, headers=headers, json=data)
    task_id = resp.json()["output"]["task_id"]

    for _ in range(30):
        r = requests.get(f"{DASHSCOPE_TASK_URL}/{task_id}", headers={"Authorization": f"Bearer {API_KEY}"})
        d = r.json()
        status = d["output"]["task_status"]
        if status == "SUCCEEDED":
            img_url = d["output"]["results"][0]["url"]
            img_resp = requests.get(img_url, timeout=30)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"{output_dir}/composite_ai_{timestamp}.png"
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            return out_path, prompt
        elif status == "FAILED":
            return None, prompt
        time.sleep(3)
    return None, prompt