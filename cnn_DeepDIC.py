import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock, ResNet
from torch.nn import init

###################################
# 1) DIC Dataset
###################################
class DICDataset(Dataset):
    def __init__(self, data_dir="generated_data"):
        super().__init__()
        self.data_dir = data_dir
        self.ref_images = []
        self.def_images = []
        self.disp_u = []
        self.disp_v = []

        ref_fnames = [f for f in os.listdir(data_dir) if f.startswith("reference_image_") and f.endswith(".png")]
        ref_fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for ref_file in ref_fnames:
            idx = ref_file.split("_")[-1].split(".")[0]
            def_file = f"deformed_image_{idx}.png"
            u_file = f"displacement_u_{idx}.npy"
            v_file = f"displacement_v_{idx}.npy"

            if (os.path.exists(os.path.join(data_dir, def_file)) and
                os.path.exists(os.path.join(data_dir, u_file)) and
                os.path.exists(os.path.join(data_dir, v_file))):
                self.ref_images.append(ref_file)
                self.def_images.append(def_file)
                self.disp_u.append(u_file)
                self.disp_v.append(v_file)

    def __len__(self):
        return len(self.ref_images)

    def __getitem__(self, idx):
        ref_path = os.path.join(self.data_dir, self.ref_images[idx])
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = ref_img.astype(np.float32) / 255.0

        def_path = os.path.join(self.data_dir, self.def_images[idx])
        def_img = cv2.imread(def_path, cv2.IMREAD_GRAYSCALE)
        def_img = def_img.astype(np.float32) / 255.0

        u_path = os.path.join(self.data_dir, self.disp_u[idx])
        v_path = os.path.join(self.data_dir, self.disp_v[idx])
        u = np.load(u_path).astype(np.float32)
        v = np.load(v_path).astype(np.float32)

        input_tensor = np.stack([ref_img, def_img], axis=0)
        label_tensor = np.stack([u, v], axis=0)

        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)

###################################
# 2) New Encoder-Decoder using ResNet
###################################

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant_(layer.bias, 0)
    return layer

def bn(planes):
    layer = nn.BatchNorm2d(planes)
    init.constant_(layer.weight, 1)
    init.constant_(layer.bias, 0)
    return layer

class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 14, 16, 3], 1000)
        self.conv_f = conv(2,64, kernel_size=3,stride = 1)
        self.ReLu_1 = nn.ReLU(inplace=True)
        self.conv_pre = conv(512, 1024, stride=2, transposed=False)
        self.bn_pre = bn(1024)

    def forward(self, x):
        x1 = self.conv_f(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.ReLu_1(self.bn_pre(self.conv_pre(x5)))
        return x1, x2, x3, x4, x5, x6

class SegResNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512,512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes, stride=2, kernel_size=5)
        init.constant_(self.conv10.weight, 0)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x + x4)))
        x = self.relu(self.bn7(self.conv7(x + x3)))
        x = self.relu(self.bn8(self.conv8(x + x2)))
        x = self.relu(self.bn9(self.conv9(x + x1)))
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x

###################################
# 3) Training Routine
###################################

def train_dic_model(data_dir="generated_data", epochs=2, batch_size=1, lr=1e-3):
    dataset = DICDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fnet = FeatureResNet()
    model = SegResNet(2, fnet).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return model

###################################
# 4) Evaluation Routine 
###################################
def evaluate_model(model, data_dir="generated_data"):
    dataset = DICDataset(data_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            pred = model(inputs)
            pred_np = pred[0].cpu().numpy()
            label_np = labels[0].cpu().numpy()

            print("Displacement prediction:", pred_np.shape, "min:", pred_np.min(), "max:", pred_np.max())
            print("Displacement label:", label_np.shape, "min:", label_np.min(), "max:", label_np.max())

            if i == 0:
                break

###################################
# 5) Example Usage
###################################
if __name__ == "__main__":
    model = train_dic_model(data_dir="generated_data", epochs=10, batch_size=1, lr=1e-3)
    evaluate_model(model, data_dir="generated_data")
    print("\nDone training and evaluation!")