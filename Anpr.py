"""
anpr.py — Minimal ANPR System (CPU-only)
Dataset: images named like MH12AB1234.jpg (filename = label)
Usage:
    python anpr.py --mode train --data_dir ./dataset
    python anpr.py --mode infer --image test.jpg --model plate_model.pth
"""

import os, sys, argparse, re
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX    = {c: i for i, c in enumerate(PLATE_CHARS)}
IDX2CHAR    = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(PLATE_CHARS)          # 36
PLATE_LEN   = 10                        # fixed length (pad/trim)
IMG_H, IMG_W = 64, 128                  # input to CNN
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3


def normalize_model_path(path):
    """Ensure model path has a .pth extension when none is provided."""
    if not path:
        return 'plate_model.pth'
    root, ext = os.path.splitext(path)
    return path if ext else f"{path}.pth"


def suggest_images(data_dir, limit=3):
    """Return a small list of sample image paths from the dataset folder."""
    if not os.path.isdir(data_dir):
        return []
    exts = ('.jpg', '.jpeg', '.png')
    samples = [
        os.path.join(data_dir, fname)
        for fname in sorted(os.listdir(data_dir))
        if fname.lower().endswith(exts)
    ]
    return samples[:limit]

# ─────────────────────────────────────────────
# 1. CLASSICAL PLATE DETECTION (OpenCV)
# ─────────────────────────────────────────────
def detect_plate(img_bgr):
    """
    Try to find a rectangular plate region using edge detection.
    Returns cropped plate (BGR) or the whole image if nothing found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:                        # rectangular shape
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 2.0 < ratio < 6.0:                   # typical plate aspect ratio
                plate = img_bgr[y:y+h, x:x+w]
                return plate

    return img_bgr                                  # fallback: full image

# ─────────────────────────────────────────────
# 2. LABEL ENCODING / DECODING
# ─────────────────────────────────────────────
def encode_label(text):
    """String → list of int indices, fixed length."""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    text = text[:PLATE_LEN].ljust(PLATE_LEN, '0')   # trim or pad with '0'
    return [CHAR2IDX[c] for c in text]

def decode_label(indices):
    """List of int indices → string."""
    return ''.join(IDX2CHAR[i] for i in indices)

# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class PlateDataset(Dataset):
    """
    Loads images from a folder.
    Label = filename without extension.
    """
    def __init__(self, folder, augment=False):
        self.samples = []
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                label_str = os.path.splitext(fname)[0]
                path = os.path.join(folder, fname)
                self.samples.append((path, label_str))

        # basic augmentation transforms
        aug_list = []
        if augment:
            aug_list += [
                T.RandomRotation(5),
                T.ColorJitter(brightness=0.3, contrast=0.3),
            ]
        self.transform = T.Compose(aug_list + [
            T.Resize((IMG_H, IMG_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label_str = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(encode_label(label_str), dtype=torch.long)
        return img, label

# ─────────────────────────────────────────────
# 4. SIMPLE CNN MODEL
# ─────────────────────────────────────────────
class PlateCNN(nn.Module):
    """
    Small CNN → flatten → one FC head per character position.
    Output: PLATE_LEN logits tensors of size NUM_CLASSES each.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # after 3 × MaxPool2d(2): H=64→8, W=128→16
        feat_dim = 128 * 8 * 16

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

        # one classifier per character position
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, NUM_CLASSES) for _ in range(PLATE_LEN)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return [head(x) for head in self.heads]  # list of PLATE_LEN tensors

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train(data_dir, save_path='plate_model.pth'):
    dataset = PlateDataset(data_dir, augment=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model   = PlateCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(dataset)} samples for {EPOCHS} epochs …")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total   = 0

        for imgs, labels in loader:
            # imgs: (B,3,H,W)  labels: (B, PLATE_LEN)
            optimizer.zero_grad()
            outputs = model(imgs)           # list of PLATE_LEN (B, NUM_CLASSES)

            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(PLATE_LEN))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # accuracy: all characters correct
            preds = torch.stack([o.argmax(1) for o in outputs], dim=1)  # (B, PLATE_LEN)
            correct += (preds == labels).all(dim=1).sum().item()
            total   += imgs.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch:>3}/{EPOCHS}  loss={total_loss/len(loader):.4f}  acc={acc:.1f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path}")

# ─────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────
def infer(image_path, model_path='plate_model.pth', data_dir='./dataset'):
    from PIL import Image

    model_path = normalize_model_path(model_path)
    if not os.path.isfile(model_path):
        print(f"Error: model file not found: {model_path}")
        print(f"Train first with: python Anpr.py --mode train --data_dir ./dataset --model {model_path}")
        return None

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        samples = suggest_images(data_dir)
        if samples:
            print("Try one of these sample images:")
            for sample in samples:
                print(f"  {sample}")
            print(f"Example: python Anpr.py --mode infer --image {samples[0]} --model {model_path}")
        return None

    # load model
    model = PlateCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    # step 1: detect plate region
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: cannot decode image file: {image_path}")
        return None

    plate_bgr = detect_plate(img_bgr)

    # step 2: convert to PIL and preprocess
    plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    plate_pil = Image.fromarray(plate_rgb)
    inp = transform(plate_pil).unsqueeze(0)          # (1,3,H,W)

    # step 3: predict
    with torch.no_grad():
        outputs = model(inp)                          # list of PLATE_LEN (1, NUM_CLASSES)
        indices = [o.argmax(1).item() for o in outputs]
        predicted = decode_label(indices)

    print(f"Detected plate: {predicted}")
    return predicted

# ─────────────────────────────────────────────
# 7. SIMPLE DUPLICATE DETECTION (string match)
# ─────────────────────────────────────────────
def find_duplicates(data_dir):
    """Print any duplicate plate numbers found in the dataset folder."""
    seen = {}
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            label = os.path.splitext(fname)[0].upper()
            seen.setdefault(label, []).append(fname)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        print("Duplicates found:")
        for plate, files in dups.items():
            print(f"  {plate}: {files}")
    else:
        print("No duplicates found.")

# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal ANPR System')
    parser.add_argument('--mode',      choices=['train', 'infer', 'duplicates'], required=True)
    parser.add_argument('--data_dir',  default='./dataset',      help='folder with plate images')
    parser.add_argument('--image',     default='test.jpg',        help='image for inference')
    parser.add_argument('--model',     default='plate_model.pth', help='model save/load path')
    args = parser.parse_args()
    model_path = normalize_model_path(args.model)

    if args.mode == 'train':
        train(args.data_dir, model_path)
    elif args.mode == 'infer':
        infer(args.image, model_path, args.data_dir)
    elif args.mode == 'duplicates':
        find_duplicates(args.data_dir)
