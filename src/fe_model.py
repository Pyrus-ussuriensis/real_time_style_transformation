import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import vgg19
#feature extracting model
fe_model = vgg19(weights="IMAGENET1K_V1").features.eval().to(device=device)
for param in fe_model.parameters():
    param.requires_grad=False

from torchvision.models.feature_extraction import create_feature_extractor 

#这里选择的层数是超参数，可以自己调节
content_layers = ['21']
style_layers = ['0', '5', '10', '19', '28']
extractor = create_feature_extractor(fe_model, return_nodes={**{l: l for l in content_layers}, **{l: l for l in style_layers}})



