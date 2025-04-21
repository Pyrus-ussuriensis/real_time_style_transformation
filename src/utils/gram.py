import torch

# 得到纹理特征的gram矩阵计算
def gram_matrix(feature):
    b, c, h, w = feature.size()
    feature = feature.view(b, c, h * w)
    return torch.bmm(feature, feature.transpose(1, 2)) / (c * h * w)