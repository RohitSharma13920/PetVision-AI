import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device}")

# 1. Locate PetImages
candidates = [
    r"C:\Users\Lenovo\python\cats_and_dogs_data\PetImages",
    r"C:\Users\Lenovo\cats_and_dogs_data\PetImages",
    "PetImages",
]

pet_dir = None
for c in candidates:
    if os.path.exists(os.path.join(c, "Cat")) and os.path.exists(os.path.join(c, "Dog")):
        pet_dir = c
        break

if not pet_dir:
    raise FileNotFoundError("PetImages folder not found!")

cat_dir = os.path.join(pet_dir, "Cat")
dog_dir = os.path.join(pet_dir, "Dog")

# 2. Balanced Split (2,000 Cats + 2,000 Dogs for training, 200 each for testing)
cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

random.seed(42)
random.shuffle(cat_files)
random.shuffle(dog_files)

train_cats, test_cats = cat_files[:2000], cat_files[2000:2200]
train_dogs, test_dogs = dog_files[:2000], dog_files[2000:2200]

train_paths = train_cats + train_dogs
train_labels = [0] * len(train_cats) + [1] * len(train_dogs)

combined = list(zip(train_paths, train_labels))
random.shuffle(combined)
train_paths, train_labels = zip(*combined)

print(f"Dataset: 4,000 Training Images | 400 Validation Images")

# 3. Robust Dataset Pipeline
class PetDataset(Dataset):
    def __init__(self, paths, lbls, augment=True):
        self.paths = paths
        self.lbls = lbls
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB").resize((128, 128))
            if self.augment and random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            tensor_img = torch.tensor((arr - mean) / std, dtype=torch.float32)
        except Exception:
            tensor_img = torch.zeros((3, 128, 128), dtype=torch.float32)

        return tensor_img, torch.tensor(self.lbls[idx], dtype=torch.long)

train_loader = DataLoader(PetDataset(train_paths, train_labels, augment=True), batch_size=32, shuffle=True)

# 4. Modern 4-Stage Deep CNN with Adaptive Global Pooling
class PetVisionCNN(nn.Module):
    def __init__(self):
        super(PetVisionCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16 -> 8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), # Forces spatial invariant focus
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = PetVisionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
epochs = 7

print("\n--- Training Deep CNN ---")
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

# 5. Live Test on Unseen Cats and Dogs
print("\n--- Model Verification on Test Set ---")
model.eval()
test_samples = [(p, 0, "Cat") for p in test_cats[:4]] + [(p, 1, "Dog") for p in test_dogs[:4]]
with torch.no_grad():
    for path, expected_lbl, label_name in test_samples:
        try:
            img = Image.open(path).convert("RGB").resize((128, 128))
            arr = (np.transpose(np.array(img, dtype=np.float32)/255.0, (2, 0, 1)) - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
            out = torch.softmax(model(t), dim=1)[0]
            pred = "Cat" if out[0] > out[1] else "Dog"
            conf = max(out[0], out[1]).item() * 100
            status = "✅ PASS" if pred == label_name else "❌ FAIL"
            print(f"Actual: {label_name:<4} | Predicted: {pred:<4} ({conf:.1f}%) | {status}")
        except Exception:
            pass

# 6. Save Model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
save_path = os.path.join(models_dir, "cat_dog_cnn.pth")
torch.save(model.state_dict(), save_path)
print(f"\nSUCCESS: Production model weights saved at: {save_path}")