import base64, requests, time

API_KEY = 'sk-e28ec115af2a4f2d95f475fb50e161e3'

with open('input/portraits/f310dc72470506fcd4637cd2ce5f5855.jpg', 'rb') as f:
    p_b64 = base64.b64encode(f.read()).decode('utf-8')
with open('input/emojis/emoji_frame.png', 'rb') as f:
    e_b64 = base64.b64encode(f.read()).decode('utf-8')

headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json', 'X-DashScope-Async': 'enable'}

prompt1 = "Transform this portrait photo into an exaggerated cartoon caricature. Keep the persons facial identity and features exactly the same, but amplify the most distinctive traits in a humorous comic style with bold outlines and vibrant colors."
data1 = {
    'model': 'wan2.7-image-pro',
    'input': {
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': f'data:image/jpeg;base64,{p_b64}'},
                {'type': 'text', 'text': prompt1}
            ]}
        ]
    },
    'parameters': {'size': '1024*1024', 'n': 1, 'prompt_extend': False}
}

prompt2 = "Combine the first portrait image with the second emoji/pet image. Keep the exact facial features and identity of the person from the first image, but render them in the cute cartoon style of the second image. The result should look like the specific person from photo 1 turned into a cute pet meme character like photo 2. Preserve identity, change style."
data2 = {
    'model': 'wan2.7-image-pro',
    'input': {
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': f'data:image/jpeg;base64,{p_b64}'},
                {'type': 'image', 'image': f'data:image/png;base64,{e_b64}'},
                {'type': 'text', 'text': prompt2}
            ]}
        ]
    },
    'parameters': {'size': '1024*1024', 'n': 1, 'prompt_extend': False}
}

r1 = requests.post('https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation', headers=headers, json=data1)
task1 = r1.json()['output']['task_id']
print(f'Caricature task: {task1}')

r2 = requests.post('https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation', headers=headers, json=data2)
task2 = r2.json()['output']['task_id']
print(f'Composite task: {task2}')

for task_id, label in [(task1, 'caricature'), (task2, 'composite')]:
    for i in range(30):
        r3 = requests.get(f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}', headers={'Authorization': f'Bearer {API_KEY}'})
        d = r3.json()
        status = d['output']['task_status']
        print(f'{label} {i+1}: {status}')
        if status == 'SUCCEEDED':
            img_url = d['output']['choices'][0]['message']['content'][0]['image']
            img_resp = requests.get(img_url, timeout=60)
            path = f'output/{label}_portrait_ref.png'
            with open(path, 'wb') as f:
                f.write(img_resp.content)
            print(f'{label}: {len(img_resp.content)} bytes -> {path}')
            break
        elif status == 'FAILED':
            print(f'{label}: FAILED')
            break
        time.sleep(5)