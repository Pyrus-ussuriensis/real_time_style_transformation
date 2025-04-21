import torch
from utils.cfg import cfg
from torchvision.models import vgg19
from utils.cfg import cfg
#feature extracting model

# 加载预训练模型权重，获取特征提取部分，转为测试模式，放到机器上，冻结参数，获取参数，根据给定层号建立特征提取器
fe_model = vgg19(weights="IMAGENET1K_V1").features.eval().to(device=cfg['device'])
for param in fe_model.parameters():
    param.requires_grad=False

from torchvision.models.feature_extraction import create_feature_extractor 
device = cfg['device']


content_layers = cfg['content_layers']
style_layers = cfg['style_layers']
extractor = create_feature_extractor(fe_model, return_nodes={**{l: l for l in content_layers}, **{l: l for l in style_layers}}).to(device=device)


