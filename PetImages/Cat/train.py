import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Locate PetImages Folder
search_paths = [
    "PetImages",
    "cats_and_dogs_data/PetImages",
    r"C:\Users\Lenovo\cats_and_dogs_data\PetImages",
    r"C:\Users\Lenovo\python\cats_and_dogs_data\PetImages",
]

pet_dir = None
for p in search_paths:
    if os.path.exists(os.path.join(p, "Cat")) and os.path.exists(os.path.join(p, "Dog")):
        pet_dir = p
        break

if not pet_dir:
    raise FileNotFoundError("Could not find 'PetImages' folder with 'Cat' and 'Dog'.")

cat_dir = os.path.join(pet_dir, "Cat")
dog_dir = os.path.join(pet_dir, "Dog")

# 3. Load Strictly Balanced Samples (1,500 Cats + 1,500 Dogs)
cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

random.seed(42)
random.shuffle(cat_files)
random.shuffle(dog_files)

num_samples = 1500
selected_cats = cat_files[:num_samples]
selected_dogs = dog_files[:num_samples]

file_paths = selected_cats + selected_dogs
labels = [0] * len(selected_cats) + [1] * len(selected_dogs)

combined = list(zip(file_paths, labels))
random.shuffle(combined)
file_paths, labels = zip(*combined)

print(f"Dataset ready: {len(file_paths)} total (1,500 Cats [0] and 1,500 Dogs [1])")

# 4. Custom Dataset with Augmentation
class PetDataset(Dataset):
    def __init__(self, paths, lbls):
        self.paths = paths
        self.lbls = lbls

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB").resize((128, 128))
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            tensor_img = torch.tensor((arr - mean) / std, dtype=torch.float32)
        except Exception:
            tensor_img = torch.zeros((3, 128, 128), dtype=torch.float32)

        return tensor_img, torch.tensor(self.lbls[idx], dtype=torch.long)

train_loader = DataLoader(PetDataset(file_paths, labels), batch_size=32, shuffle=True)

# 5. 3-Block CNN Model
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
epochs = 6

print("\n--- Training Started ---")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = (correct / total) * 100
    loss_val = running_loss / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss_val:.4f} - Accuracy: {acc:.2f}%")

# 6. Save Weights
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
save_path = os.path.join(models_dir, "cat_dog_cnn.pth")
torch.save(model.state_dict(), save_path)
print(f"\nSUCCESS: Fresh model weights saved at: {save_path}")