# 实时风格转化
## 效果预览
![alt text](<./docs/origin.jpg>)
![alt text](<./docs/starrynight.jpg>)
![alt text](<./docs/ndnie.jpg>)
![alt text](<./docs/shennaichuan.jpg>)  

图片来源是[视频](https://pixabay.com/videos/wave-rays-intro-colorful-particles-217488/?utm_source=chatgpt.com)第一帧。这三个风格图片分别是starrynight, ndnie, 神奈川沖浪裏。
## 环境配置
根据configs/cv.yml配置环境
```bash
conda env create -f cv.yml
```
## 实时转化
``` yaml
cv_mode: 'video' # video是处理视频模式，调用visualize.py；camera是使用摄像头，但是wsl不支持
video_path: 'data/video/hfut.mp4' # 原视频路径
output_path: 'results/video/output_2.mp4' # 输出视频路径
weight_path: 'weights/rtst_4.pth' # 加载权重文件的路径
```
调节这几个参数，然后运行src/visualize.py
## 训练
```yaml
device: auto   # 可选值：auto、cuda、cpu
experiment: 4 # 训练实验号，影响部分文件名字，比如日志文件，Tensorboard文件
load: 3 # 加载权重标号，前面是RTST_，下一个是保存的标号
save: 3
a: 1 #1 # 三种损失的比例
b: 100000 #1e5
c: 1e-7 #5e-6
style_pic: 'data/impression.jpg' # 风格图片路径
content_layers: ['15']               # relu3_3
style_layers: ['3', '8', '15', '22']  # relu1_2, relu2_2, relu3_3, relu4_3
weights_path: 'weights/' # 权重文件夹
mean: [0.485, 0.456, 0.406] # imagenet常用数值
std: [0.229, 0.224, 0.225]
pic_size: 256 # 输入图片裁切尺寸

lr: 1e-3            # 学习率  
epochs: 20          # 迭代次数  
batch_size: 16
freq: 50 # 多少步打印一个Batchsize的图片
train_subset_len: 20000       # 划分数据集子集大小              
val_subset_len: 1000                    
mode: 'train' # 模式，test是使用data下的test文件夹中的一个图片，对一个图片做过拟合尝试，还有一些参数都会在src/utils/cfg.py中自动做修改；train是子集下训练；full_train是在完整数据集下训练
```
超参数已经调节好，单纯改变风格图片则只需改style_pic,load,save,mode，如果load所在权重存在，重新训练要么删除要么换新号，或者改train.py，注释load_model代码。一般train模式下训练两个epoch效果就还可以了。然后根据自己的显卡性能可以调节batchsize。然后运行train.py.
## 参考文献
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)  
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution: SupplementaryMaterial](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)