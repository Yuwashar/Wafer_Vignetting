import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))



class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResUNet, self).__init__()

        self.e1 = ResBlock(in_channels, 64);
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(64, 128);
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(128, 256);
        self.pool3 = nn.MaxPool2d(2)
        self.b = ResBlock(256, 512)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1 = ResBlock(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2 = ResBlock(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3 = ResBlock(128 + 64, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.e1(x);
        p1 = self.pool1(c1)
        c2 = self.e2(p1);
        p2 = self.pool2(c2)
        c3 = self.e3(p2);
        p3 = self.pool3(c3)
        b = self.b(p3)

        u1 = self.up1(b);
        cat1 = torch.cat((c3, u1), dim=1);
        d1 = self.d1(cat1)
        u2 = self.up2(d1);
        cat2 = torch.cat((c2, u2), dim=1);
        d2 = self.d2(cat2)
        u3 = self.up3(d2);
        cat3 = torch.cat((c1, u3), dim=1);
        d3 = self.d3(cat3)

        return self.sigmoid(self.output(d3))



class WaferDataset(Dataset):
    def __init__(self, input_dir, target_dir, num_files=88, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        # 输入是 JPG，标签是 PNG
        self.images = [f"{i}.jpg" for i in range(num_files)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name_jpg = self.images[idx]
        img_name_png = img_name_jpg.replace('.jpg', '.png')

        # 输入：JPG 
        input_path = os.path.join(self.input_dir, img_name_jpg)
        # 标签：PNG
        target_path = os.path.join(self.target_dir, img_name_png)

        try:
            input_img = Image.open(input_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
        except FileNotFoundError as e:

            raise FileNotFoundError(f"文件未找到：请确保 JPG/PNG 文件都已存在于相应路径。{e}")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img



def main():
    INPUT_PATH = "/tmp/pycharm_project_468/data/data/raw_for_dl"
    TARGET_PATH = "/tmp/pycharm_project_468/data/data/label_for_dl/label_for_dl_png"

    IMAGE_SIZE = 512

    NUM_FILES = 88
    BATCH_SIZE = 4
    LR = 0.001
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {DEVICE}")
    print(f"训练图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = WaferDataset(INPUT_PATH, TARGET_PATH, NUM_FILES, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    model = ResUNet(in_channels=3, out_channels=3).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"数据集大小: {len(train_dataset)} 张图, 共 {len(train_loader)} 批次")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss (MSE): {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            save_path = f"/tmp/resunet_restore_{IMAGE_SIZE}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
