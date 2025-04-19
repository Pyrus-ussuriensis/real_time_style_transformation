from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

class StyleDataset(Dataset):
    def __init__(self, root):
        self.pics = glob.glob(f"{root}/*.jpg")
        self.tf = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)

        ])
    def __len__(self): return len(self.pics)
    def __getitem__(self, i):
        img = Image.open(self.pics[i]).convert('RGB')
        return self.tf(img)

TrainLoader = DataLoader(StyleDataset("data/train"), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
ValidateLoader = DataLoader(StyleDataset("data/val"), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

