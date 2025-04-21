# learning notes
## 整理数据集
```bash
cd /path/to/dir
find . -mindepth 2 -type f -exec mv -t . -- {} +
find . -type d -mindepth 1 -empty -delete
```
首先到达图片所在的文件夹，然后这两句分别是将子文件中的内容放到主文件夹中，然后将所有空文件夹都删除，这是因为我所用的数据集IMAGENET2012的结构是这样的。我直接选了前100个文件夹作为训练集，大概十几万张图片，然后挑选了5个文件夹作为验证集。

## 项目结构
```bash
project/
├── data/           # 原始与预处理数据
├── notebooks/      # Jupyter 实验笔记
├── src/            # 核心代码（模型、训练、评估模块）
├── configs/        # 超参与运行配置
├── models/         # 网络定义
├── weights/        # 检查点／权重
├── logs/           # 终端日志
├── tensorboard/    # 可视化记录
├── utils/          # 公共工具函数
├── results/        # 输出（图表、报告）
└── docs/           # 文档／设计笔记
```
* data 数据集要放到外面用软连接使用，不然git扫描会很慢
  ```bash
  ln -s /mnt/d/datasets/mydata ./data
  ls -l data  # 确认为指向外部的符号链接
  ```
* notebooks 放置对于实验中的交互代码的快速实验
* src
  opencv
  ```python
  import cv2, torch
  from torchvision import transforms
  from PIL import Image

  # 加载训练好的TorchScript模型
  model = torch.jit.load("style_transformer.pt").to(device).eval()

  # 推理时只需Resize→ToTensor→Normalize
  infer_tf = transforms.Compose([
      transforms.Resize((512,512)),
      transforms.ToTensor(),           
      transforms.Normalize(mean,std)
  ])

  cap = cv2.VideoCapture(0)
  while True:
      ret, frame = cap.read()
      if not ret: break
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      t = infer_tf(Image.fromarray(img)).unsqueeze(0).to(device)
      with torch.no_grad():
          out = model(t).cpu().squeeze(0)
      # 反归一化并转回BGR
      out = (out.clamp(0,1).permute(1,2,0)*255).byte().numpy()
      out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
      cv2.imshow("Stylized", out)
      if cv2.waitKey(1) == 27: break
  cap.release()
  cv2.destroyAllWindows()
  ```
* configs 将网络的超参数配置从python文件中分离，放在yaml文件中，后期导入
  ```bash
  conda install -c conda-forge pyyaml
  ```
  写入格式
  ```python
  train:
    lr: 3e-4            # 学习率  
    batch_size: 32      # 批大小  
    epochs: 50          # 迭代次数  

  paths:
    data: "../data"     # 数据目录  
    save: "./weights"   # 权重输出目录  

  model:
    depth: 50           # 网络深度  
    dropout: 0.5        # 随机失活率  

  # 列表示例
  augmentations:
    - flip
    - rotate
    - color_jitter
  ```
  缩进：统一用 2 个空格，不要用 Tab
  注释：# 后跟内容
  键值：key: value；字符串可加引号，也可省略
  列表：- item 形式

  读取
  ```python
  # 参数与配置：argparse + yaml
  import argparse, yaml

  p = argparse.ArgumentParser()
  p.add_argument('--cfg', required=True, help='配置文件路径')
  args = p.parse_args()
  cfg = yaml.safe_load(open(args.cfg, 'r'))
  # 访问示例
  lr = cfg['train']['lr']
  batch = cfg['train']['batch_size']
  ```
* models 网络架构从python文件中分离
* logs 使用
  ```python
  # 日志记录：RotatingFileHandler 自动轮转
  import logging
  from logging.handlers import RotatingFileHandler

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  # 文件最大 10MB，保留 5 个备份
  handler = RotatingFileHandler('logs/train.log', maxBytes=10*1024*1024, backupCount=5)
  logger.addHandler(handler)
  logger.addHandler(logging.StreamHandler())
  logger.info('训练开始')
  ```
* tensorboard 
  ```python
  from torch.utils.tensorboard import SummaryWriter

  writer = SummaryWriter('logs/tb')
  writer.add_scalar('train/loss', loss, step)
  ```
* results 放置输出的图片，PDF之类
* docs 架构说明，算法推导，决策
* .gitignore 忽略比较沉重的或者可以复现的数据结果等内容
  ```bash
  # 数据、输出、缓存
  data/           # 原始/预处理数据  
  weights/        # 模型检查点  
  logs/           # 文本日志  
  tensorboard/    # TB 记录  
  results/        # 实验输出  
  __pycache__/    # Python 缓存  
  *.pyc           # 字节码  
  .vscode/        # 编辑器配置  
  .env            # 环境变量  
  ```

## 项目规划
1. 完整基本的项目架构规划
   * .vscode 
   * configs 
     - [x] rtst.yaml 基本配置初始化
   * data
     - [x] 数据集准备
     - [x] 建立数据集类，做裁切，索引
   * models
     - [x] 建立模型架构rtst
   * notebooks
     - [x] analysis.ipynb记录分析初始化
   * src
     - [x] vgg提取
2. 初始完成
   - [x] 完成训练准备
     - [x] 记录日志
     - [x] 实验笔记结果
     - [x] Tensorboard
     - [x] 权重文件
     - [x] 完成损失函数设计
     - [x] 可视化处理结果
     - [x] 完成训练，测试代码
   - [x] 测试使用
     - [x] opencv
3. 细节优化