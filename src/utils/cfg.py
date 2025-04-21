import yaml
import torch
from PIL import Image
from torchvision import transforms

# 在这里统一读取参数，放到一个字典中，如果要读取参数，导入这个文件读取
with open("configs/rtst.yaml","r",encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 一些其他参数可以在这里统一加入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg['device'] = device
style_pic = cfg['style_pic']
style_pic = Image.open(style_pic).convert('RGB')
mean = cfg['mean']
std = cfg['std']
pic_size = int(cfg['pic_size'])
transform_pic = transforms.Compose([
            transforms.Resize(pic_size),
            transforms.CenterCrop(pic_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)

        ])

transform_pics = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)

        ])
style_pic = transform_pic(style_pic)

cfg['style_pic'] = style_pic.unsqueeze(0).to(device)

# 根据模式修改部分参数
mode = cfg['mode']
if mode == 'test':
    cfg['epochs'] = 2000          # 迭代次数  
    cfg['batch_size'] = 1
    cfg['freq'] = 5