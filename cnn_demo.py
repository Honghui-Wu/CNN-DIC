import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

###################################
# 1) DIC Dataset
###################################
class DICDataset(Dataset):
    def __init__(self, data_dir="generated_data"):
        super().__init__()
        self.data_dir = data_dir
        # Gather all samples based on saved filenames
        self.ref_images = []
        self.def_images = []
        self.disp_u = []
        self.disp_v = []

        # Identify how many samples we have by listing reference images
        # e.g. reference_image_0.png
        ref_fnames = [f for f in os.listdir(data_dir) if f.startswith("reference_image_") and f.endswith(".png")]
        # Sort to ensure consistent ordering
        ref_fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        #print("ref_fnames_sorted", ref_fnames)

        for ref_file in ref_fnames:
            # Extract index from filename
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
                #print("self.ref_images",self.ref_images)

    def __len__(self):
        return len(self.ref_images)

    def __getitem__(self, idx):
        # Load reference image
        ref_path = os.path.join(self.data_dir, self.ref_images[idx])
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = ref_img.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Load deformed image
        def_path = os.path.join(self.data_dir, self.def_images[idx])
        def_img = cv2.imread(def_path, cv2.IMREAD_GRAYSCALE)
        def_img = def_img.astype(np.float32) / 255.0  # Normalize

        # Load displacement fields
        u_path = os.path.join(self.data_dir, self.disp_u[idx])
        v_path = os.path.join(self.data_dir, self.disp_v[idx])
        u = np.load(u_path).astype(np.float32)
        v = np.load(v_path).astype(np.float32)

        # Stack reference & deformed images in the channel dimension => shape: (2,H,W)
        # Then stack displacement fields => shape: (2,H,W)
        input_tensor = np.stack([ref_img, def_img], axis=0)  # (2, H, W)
        label_tensor = np.stack([u, v], axis=0)              # (2, H, W)

        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)

###################################
# 2) Encoder-Decoder CNN Model
###################################

class SimpleEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: (2,H,W) -> latent
        # We'll do a small model for demonstration
        # in_channels=2, out_channels=8
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Decoder: latent -> (2,H,W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec

###################################
# 3) Training Routine
###################################

def train_dic_model(data_dir="generated_data", epochs=2, batch_size=1, lr=1e-3):
    # Create dataset & dataloader
    dataset = DICDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEncoderDecoder().to(device)

    # Loss function (e.g., MSE for displacement regression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            # inputs: (B,2,H,W), labels: (B,2,H,W)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            preds = model(inputs)

            # Compute loss
            loss = criterion(preds, labels)

            # Backward pass
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
    """
    Evaluates the trained model on a test dataset.
    """
    dataset = DICDataset(data_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            pred = model(inputs)
            # inputs shape: (1, 2, H, W), labels shape: (1, 2, H, W)
            # pred shape: (1, 2, H, W)

            # Convert predictions and label to numpy for visualization
            pred_np = pred[0].cpu().numpy()
            label_np = labels[0].cpu().numpy()

            # We can do some quick numeric check or partial visualization.
            # For brevity, just print min/max for first sample
            print("Displacement prediction:", pred_np.shape, "min:", pred_np.min(), "max:", pred_np.max())
            print("Displacement label:", label_np.shape, "min:", label_np.min(), "max:", label_np.max())

            # Only check the first sample
            if i == 0:
                break

###################################
# 5) Example Usage
###################################
if __name__ == "__main__":
    # Train the model on the generated DIC data
    model = train_dic_model(data_dir="generated_data", epochs=100, batch_size=1, lr=1e-3)

    # Evaluate the trained model
    evaluate_model(model, data_dir="generated_data")

    print("\nDone training and evaluation!")