import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

from model import CSRNet

DATA_DIR = os.path.join("..", "dataset", "train")
EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-5

#Dataset pro CSRNet
class CSRNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [img for img in os.lisdir(data_dir) if img.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.data_dir, img_name)
        npy_path = os.path.join(self.data_dir, img_name.replace("jpg", ".npy"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        density = np.load(npy_path)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #são as médias e desvio padrão dos canais R, G, B no ImageNet, respectivamente
])

dataset = CSRNetDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Modelo
model = CSRNet().cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)

#treinamento

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for imgs, densities in tqdm(dataloader, desc=f"Epoch{epoch+1}/{EPOCHS}"):
        imgs, densities = imgs.cuda(), densities.cuda()

        outputs = model(imgs),
        loss = criterion(outputs, densities)    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss =+ loss.item()
    
    print(f"Epoch {epoch+1} Loss:{epoch_loss/len(dataloader):.4f}")

#salva o modelo
torch.save(model.state_dict(), "csrnet_weights.pth")
print("Modelo CSRNet treinado e salvo com sucesso!")