import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.rtst import TransformerNet
from src.utils.cfg import cfg, transform_pic, transform_pics

cv_mode = cfg['cv_mode']
mean = cfg['mean']
std = cfg['mean']


# 初始化参数设备和模型
device = cfg['device']
video_path = cfg['video_path']
weight_path = cfg['weight_path']
output_path = cfg['output_path']

print(device)
model = TransformerNet().to(device).eval()
checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)

# 后处理：(-1,1)->(0,1) 并转回 NumPy
def postprocess(tensor):
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()  # H×W×C
    return (img * 255).astype(np.uint8)

# 打开视频文件

if cv_mode == 'video':
    # 处理视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # mp4v/mjpg/xvid
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    delay = int(1000 / fps)
elif cv_mode == 'camera':
    # 使用摄像头
    cap = cv2.VideoCapture(0)
    delay = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理：BGR->RGB->PIL->Tensor
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    input_tensor = transform_pics(pil).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]

    # 后处理 + 转回 BGR
    out_img = postprocess(output_tensor)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    out.write(out_bgr)

    # 显示与退出判断
    cv2.imshow("Stylized", out_bgr)
    key = cv2.waitKey(delay=delay)
    if key == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
