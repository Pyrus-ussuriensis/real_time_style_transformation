from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import glob
from src.utils.cfg import cfg

# 调用参数，归一化的参数，图片尺寸，当前模式
mean = cfg['mean']
std = cfg['std']
batch_size = cfg['batch_size']
pic_size = int(cfg['pic_size'])
mode = cfg['mode']

class StyleDataset(Dataset):
    def __init__(self, root):
        self.pics = glob.glob(f"{root}/*.JPEG")
        self.tf = transforms.Compose([
            transforms.Resize(pic_size),
            transforms.CenterCrop(pic_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)

        ])
    def __len__(self): return len(self.pics)
    def __getitem__(self, i):
        img = Image.open(self.pics[i]).convert('RGB')
        return self.tf(img)

train_subset_len = int(cfg['train_subset_len'])
val_subset_len = int(cfg['val_subset_len'])

# 根据模式确定要加载的数据集
if mode == 'train':
    train_full = StyleDataset("data/train")
    train_ds, _ = random_split(train_full, [train_subset_len, len(train_full)-train_subset_len])

    val_full = StyleDataset("data/val")
    val_ds, _ = random_split(val_full, [val_subset_len, len(val_full)-val_subset_len])


    TrainLoader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    ValidateLoader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif mode == 'test':
    train_full = StyleDataset("data/test")
    TrainLoader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    ValidateLoader = None
elif mode == 'full_train':
    train_full = StyleDataset("data/train")

    val_full = StyleDataset("data/val")


    TrainLoader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    ValidateLoader = DataLoader(val_full, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

