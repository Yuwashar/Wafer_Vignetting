import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


# --- ResBlock Definition ---
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


# --- ResUNet Model Definition ---
class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResUNet, self).__init__()
        self.e1 = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(128, 256)
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
        c1 = self.e1(x)
        p1 = self.pool1(c1)
        c2 = self.e2(p1)
        p2 = self.pool2(c2)
        c3 = self.e3(p2)
        p3 = self.pool3(c3)
        b = self.b(p3)
        u1 = self.up1(b)
        cat1 = torch.cat((c3, u1), dim=1)
        d1 = self.d1(cat1)
        u2 = self.up2(d1)
        cat2 = torch.cat((c2, u2), dim=1)
        d2 = self.d2(cat2)
        u3 = self.up3(d2)
        cat3 = torch.cat((c1, u3), dim=1)
        d3 = self.d3(cat3)
        return self.sigmoid(self.output(d3))


# --- WaferDataset Definition ---
class WaferDataset(Dataset):
    def __init__(self, input_dir, target_dir, num_files, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = [f"{i}.jpg" for i in range(num_files)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name_jpg = self.images[idx]
        img_name_png = img_name_jpg.replace('.jpg', '.png')

        input_path = os.path.join(self.input_dir, img_name_jpg)
        target_path = os.path.join(self.target_dir, img_name_png)

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img


# --- Evaluation Logic ---
def evaluate_model():
    # --- Configuration ---
    TEST_INPUT_PATH = "./data/test_raw"
    TEST_TARGET_PATH = "./data/test_label"
    MODEL_WEIGHTS_FILE = "resunet_restore_512_epoch_50.pth"

    NUM_TEST_FILES = 88
    IMAGE_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Evaluation Start: Loading model {MODEL_WEIGHTS_FILE} ---")

    # Data Loader
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    test_dataset = WaferDataset(TEST_INPUT_PATH, TEST_TARGET_PATH, NUM_TEST_FILES, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model Instantiation and Weight Loading
    model = ResUNet(in_channels=3, out_channels=3).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model weight file not found at {MODEL_WEIGHTS_FILE}.")
        return

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            output_np = outputs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            target_np = targets.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            current_psnr = peak_signal_noise_ratio(target_np, output_np, data_range=1.0)
            current_ssim = structural_similarity(target_np, output_np, data_range=1.0, channel_axis=-1,
                                                 multichannel=True)

            psnr_scores.append(current_psnr)
            ssim_scores.append(current_ssim)

    # Results Summary
    if psnr_scores:
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        print("\n==============================================")
        print("Final Evaluation Results")
        print(f"Test Samples: {len(psnr_scores)}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("==============================================")

        # Plot PSNR Distribution
        plt.figure(figsize=(10, 4))
        plt.plot(psnr_scores)
        plt.axhline(avg_psnr, color='r', linestyle='--', label=f'Avg PSNR: {avg_psnr:.2f} dB')
        plt.title('PSNR Distribution on Test Set')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR (dB)')
        plt.savefig('psnr_distribution.png')
        print("PSNR distribution plot saved to psnr_distribution.png")


if __name__ == "__main__":
    evaluate_model()