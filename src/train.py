import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from fe_model import extractor
from data.StyleDataset import TrainLoader, ValidateLoader
from models.rtst import TransformerNet
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics import TotalVariation
from utils.log import log_info
from utils.cfg import cfg, transform_pic
from utils.weights import load_model, save_model
from utils.gram import gram_matrix
from utils.save import save_pics
from src.utils.tensorboard import writer
import logging
import os

# 加载参数
RTST = TransformerNet()
device = cfg['device']
epochs = int(cfg['epochs'])
style_pic = cfg['style_pic']
freq = int(cfg['freq'])

mean = cfg['mean']
std = cfg['std']

a = float(cfg['a'])
b = float(cfg['b'])
c = float(cfg['c'])

content_layers = cfg['content_layers']
style_layers = cfg['style_layers']


lr = float(cfg['lr'])

# 构建优化器，损失函数，第二个是平滑度的损失函数，pic_num是输出图片的标号
optimizer = Adam(params=RTST.parameters(), lr=lr)
loss_func = F.mse_loss
loss_tv = TotalVariation(reduction='sum').to(device=device) # Total Variation Loss
pic_num = 0

def train(
    RTST: nn.Module,
    FE:nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    loss_tv,
    device: torch.device,
    content_layers,
    style_layers,
    style_pic,
    epochs: int,
    scheduler,
    log_fn,
    a,
    b,
    c,
    freq):
    # 转移网络到设备，提取风格图特征，仅一次
    RTST.to(device)
    with torch.no_grad():
        styles_feats = FE(style_pic)
    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs+1):
        # 网络转训练模式
        RTST.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # 处理图片计算损失
            inputs = batch.to(device)
            LT = 0 

            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            pic_gen = RTST(inputs)
            pic_save = pic_gen.detach().cpu()
            pic_gen = transforms.Normalize(mean=mean,std=std)((pic_gen+1)/2)
            Ltv = loss_tv(pic_gen) # Total Variation Loss

            with torch.no_grad():
                inputs_feats = FE(inputs)
            pic_feats = FE(pic_gen)
            Lc = 0 # Content Loss
            for layer in content_layers:
                Lc += loss_fn(pic_feats[layer], inputs_feats[layer])
            Ls = 0 # Style Loss
            for layer in style_layers:
                Ls += loss_fn(gram_matrix(pic_feats[layer]), gram_matrix(styles_feats[layer]))
            LT = a*Lc + b*Ls + c*Ltv # Total Loss

            # 隔50步记录日志，打印图片
            if step % freq == 0:
                log_fn(epoch=step, loss=LT, mode='train', place='step')
                global pic_num 
                #save_pics(inputs, pic_num, "origin")
                n = save_pics(pic_save, pic_num, "generate")
                pic_num += n

            # 优化
            LT.backward()
            torch.nn.utils.clip_grad_norm_(RTST.parameters(), 5.0)
            optimizer.step()

            '''
            scaler.scale(LT).backward()
            scaler.step(optimizer)
            scaler.update()
            '''

        if scheduler: 
            scheduler.step()
        # 记录日志，保存权重，验证类似，但不用优化
        log_fn(epoch=epoch, loss=LT.item(), mode='train', place='epoch')
        save_model(RTST, save)
        if val_loader:
            RTST.eval()
            with torch.no_grad():
                total_loss = 0
                for batch in tqdm(val_loader):
                    
                    inputs = batch.to(device)
                    LT = 0 

                    pic_gen = RTST(inputs)
                    Ltv = loss_tv(pic_gen) # Total Variation Loss

                    with torch.no_grad():
                        inputs_feats = FE(inputs)
                        pic_feats = FE(pic_gen)
                        styles_feats = FE(style_pic)
                    Lc = 0 # Content Loss
                    for layer in content_layers:
                        Lc += loss_fn(pic_feats[layer], inputs_feats[layer])
                    Ls = 0 # Style Loss
                    for layer in style_layers:
                        Ls += loss_fn(gram_matrix(pic_feats[layer]), gram_matrix(styles_feats[layer]))
                    LT = (a*Lc + b*Ls + c*Ltv).item() # Total Loss

                    total_loss += LT
                log_fn(epoch=epoch, loss=LT/len(val_loader), mode='val', place='epoch')

# 加载权重，训练，保存权重
load = int(cfg['load'])
save = int(cfg['save'])

if os.path.isfile(os.path.join('weights/', f'RTST_{load}.pth')):
    load_model(RTST, load)
train(RTST=RTST, FE=extractor, train_loader=TrainLoader, val_loader=ValidateLoader, optimizer=optimizer, loss_fn=loss_func, loss_tv=loss_tv, device=device,
      content_layers=content_layers, style_layers=style_layers, style_pic=style_pic, epochs=epochs, log_fn=log_info, a=a, b=b, c=c,scheduler=None, freq=freq)
save_model(RTST, save)

# 停止日志和Tensorboard的记录               
logging.shutdown()
writer.close()



