# 模型使用心得 & 避坑指南

## qwen3-max (视觉理解模型)

### 规则1：必须用英文system prompt
- 中文prompt → 模型倾向于给**通用模板**而非实际描述
- 加 `"role": "system", "content": "You are a visual analysis assistant that can see and describe images."` 后模型才会真正看图描述

### 规则2：要求英文输出
- 让模型直接输出英文prompt，避免中文→英文翻译的信息损失
- 示例：`"Describe the specific facial features you see... Only output the English prompt."`

### 规则3：不要用模板式提问
- ❌ "请描述面部特征（脸型、眼睛、鼻子...）" → 得到通用列表
- ✅ "Describe the specific facial features you see in THIS portrait photo" → 得到具体描述

---

## wan2.7-image-pro (图片生成模型)

### 规则1：正确API端点
- ❌ `/api/v1/services/aigc/text2image/image-synthesis` (旧版，仅文生图)
- ✅ `/api/v1/services/aigc/image-generation/generation` (新版，支持图生图+编辑)
- 必须设置 `X-DashScope-Async: enable` header

### 规则2：图片参考的权重问题
- **只传文字prompt** → 生成通用图片，与人像/表情包无关
- **只传人像图片** → 保留人像身份，但无法还原特定表情包外观
- **只传表情包图片** → 保留表情包外观，但面部变成通用猫脸
- **同时传人像+表情包** → 表情包的视觉特征**太强势**，尤其大圆眼睛会覆盖人像特征

### 规则3：表情包大眼是顽固问题
- 表情包/宠物的大圆白眼圈是最难克服的视觉干扰
- prompt中加 "CRITICAL RULE"、"NOT big round cat eyes" 等强约束**效果有限**
- ✅ 正确方案：**两步法**
  - Step1：传人像+表情包，生成初步合成（结构OK，但眼睛是圆的）
  - Step2：局部重绘修复——传合成图+眼部mask+人像参考，只重绘眼睛区域

### 规则4：人像结构 + 表情包纹理/色彩
- prompt应明确："Keep the persons facial STRUCTURE (eye shape, nose shape...) but render using the cats TEXTURE and COLOR style"
- "The persons deep-set eyes become deep-set orange fur-textured eyes"
- 这样模型会把人像的几何特征用表情包的材质来渲染

### 规则5：prompt_extend=False
- 开启时模型会自动扩展prompt，可能改变原始意图
- 关闭后更精确遵循指令

---

## wanx-v1 (旧版文生图模型)

### 结论：不适合此项目
- 仅支持文生图，不支持图片参考输入
- 生成的图片与实际人像/表情包毫无关系
- 已被 wan2.7-image-pro 完全替代

---

## 最佳工作流 (已验证有效)

```
1. qwen3-max 分析人像特征 → 生成英文caricature prompt
2. wan2.7 传入人像图片 → 生成漫画化人像 ✅ (效果OK)
3. wan2.7 传入人像+表情包 → 生成初步合成 (人像结构+表情包外观，但眼睛是圆的)
4. 创建眼部mask → 定位眼睛区域
5. wan2.7 传入合成图+mask+人像 → 局部重绘修复眼睛 ✅ (最终效果OK)
```

## 眼部mask参数
- 左眼中心: (0.35*w, 0.28*h)
- 右眼中心: (0.65*w, 0.28*h)
- 半径: min(w,h) * 0.10
- GaussianBlur (21,21) 做边缘柔化