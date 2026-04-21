import os
import random
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0

ROOT_DIR = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\data\GTSRB"
TRAIN_DIR = os.path.join(ROOT_DIR, "Final_Training", "Images")
TEST_DIR = os.path.join(ROOT_DIR, "Final_Test", "Images")
TEST_CSV = os.path.join(ROOT_DIR, "GT-final_test.csv")

SYNTH_DIR = os.path.join(ROOT_DIR, "synthetic_train_config3")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)

SELECTED_CLASSES = [12, 13, 14, 17, 25, 38]

CLASS_NAMES = {
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    17: "No entry",
    25: "Road work",
    38: "Keep right"
}

label_map = {cls_id: idx for idx, cls_id in enumerate(SELECTED_CLASSES)}
inv_label_map = {idx: cls_id for cls_id, idx in label_map.items()}


# Synthesized-data experiment: Here we are generating additional synthetic training images
#  from the original train split and train ResNet50 on real + synthetic data.



def set_seed(seed=42):
    #     Setting random seeds for reproducibility so training and synthetic image generation are repeatable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))


def build_train_dataframe(train_dir, selected_classes, label_map):
     #   Building a dataframe of real training images for the selected classes.
    records = []
    for cls_id in selected_classes:
        class_folder = os.path.join(train_dir, f"{cls_id:05d}")
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Missing training folder: {class_folder}")
        for file_name in os.listdir(class_folder):
            if file_name.lower().endswith(".ppm"):
                records.append({
                    "filepath": os.path.join(class_folder, file_name),
                    "label_original": cls_id,
                    "label": label_map[cls_id],
                    "is_synthetic": 0
                })
    return pd.DataFrame(records)


def build_test_dataframe(test_dir, test_csv_path, selected_classes, label_map):
    #   Here we Build a dataframe of filtered test images using the official GTSRB test CSV.
    test_csv = pd.read_csv(test_csv_path, sep=';')
    filtered = test_csv[test_csv["ClassId"].isin(selected_classes)].copy()
    filtered["filepath"] = filtered["Filename"].apply(lambda x: os.path.join(test_dir, x))
    filtered["label_original"] = filtered["ClassId"]
    filtered["label"] = filtered["ClassId"].map(label_map)
    filtered["is_synthetic"] = 0
    return filtered[["filepath", "label_original", "label", "is_synthetic"]].reset_index(drop=True)


def add_gaussian_noise(pil_img, sigma=18):
    #    We need to Add Gaussian noise to an image to simulate sensor noise or low-quality image capture.
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_occlusion(pil_img):
    #    We also added a random rectangular occlusion block to simulate partial obstruction of a traffic sign.
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    occ_w = int(w * random.uniform(0.18, 0.30))
    occ_h = int(h * random.uniform(0.18, 0.30))

    x1 = random.randint(0, max(0, w - occ_w))
    y1 = random.randint(0, max(0, h - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h

    color = tuple(np.random.randint(0, 256, size=3).tolist())
    draw.rectangle([x1, y1, x2, y2], fill=color)
    return img


def synthesize_image(img):
    """
    This Creates one synthetic image by applying one or two stronger saved perturbations.
    This is different from Configuration 2 because these images are generated and
    saved as new training files instead of being augmented only on the fly.
    """
    out = img.copy()

    operations = [
        "blur",
        "bright",
        "dark",
        "noise",
        "occlusion"
    ]

    chosen = random.sample(operations, k=random.choice([1, 2]))

    for op in chosen:
        if op == "blur":
            out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2)))
        elif op == "bright":
            factor = random.uniform(1.2, 1.6)
            out = ImageEnhance.Brightness(out).enhance(factor)
        elif op == "dark":
            factor = random.uniform(0.45, 0.8)
            out = ImageEnhance.Brightness(out).enhance(factor)
        elif op == "noise":
            out = add_gaussian_noise(out, sigma=random.uniform(10, 22))
        elif op == "occlusion":
            out = add_occlusion(out)

    return out


def generate_synthetic_dataset(train_split_df, synth_dir, variants_per_image=1):
    """
    Generating and saving synthetic images only from TRAIN split.
    Validation/test stay untouched.
    """
    synth_records = []

    for cls_id in SELECTED_CLASSES:
        os.makedirs(os.path.join(synth_dir, f"{cls_id:05d}"), exist_ok=True)

    print("\nGenerating synthetic training images...")

    for idx, row in train_split_df.iterrows():
        original_path = row["filepath"]
        cls_id = int(row["label_original"])
        mapped_label = int(row["label"])

        img = Image.open(original_path).convert("RGB")

        base_name = os.path.splitext(os.path.basename(original_path))[0]

        for v in range(variants_per_image):
            synth_img = synthesize_image(img)

            save_name = f"{base_name}_synth_{v}.png"
            save_path = os.path.join(synth_dir, f"{cls_id:05d}", save_name)
            synth_img.save(save_path)

            synth_records.append({
                "filepath": save_path,
                "label_original": cls_id,
                "label": mapped_label,
                "is_synthetic": 1
            })

        if (idx + 1) % 500 == 0:
            print(f"Generated synthetic images for {idx+1} training samples")

    synth_df = pd.DataFrame(synth_records)
    print("Synthetic dataset size:", len(synth_df))
    return synth_df


train_df = build_train_dataframe(TRAIN_DIR, SELECTED_CLASSES, label_map)
test_df = build_test_dataframe(TEST_DIR, TEST_CSV, SELECTED_CLASSES, label_map)

train_split_df, val_split_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED
)

train_split_df = train_split_df.reset_index(drop=True)
val_split_df = val_split_df.reset_index(drop=True)

print("\nOriginal data sizes")
print("Train:", len(train_split_df))
print("Val  :", len(val_split_df))
print("Test :", len(test_df))

# generting synthetic image per real training image
synth_df = generate_synthetic_dataset(train_split_df, SYNTH_DIR, variants_per_image=1)

combined_train_df = pd.concat([train_split_df, synth_df], ignore_index=True)
combined_train_df = combined_train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("\nCombined training size (real + synthetic):", len(combined_train_df))
print("Real train images:", len(train_split_df))
print("Synthetic train images:", len(synth_df))


class GTSRBSubsetDataset(Dataset):
    #   creating Custom dataset for loading both real and synthetic training images from a dataframe.
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# No augmentation done here — this config is "real data + synthesized data"
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset = GTSRBSubsetDataset(combined_train_df, transform=train_transform)
val_dataset = GTSRBSubsetDataset(val_split_df, transform=val_test_transform)
test_dataset = GTSRBSubsetDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def build_model(num_classes):
    #     Loading an ImageNet-pretrained ResNet50, freeze the feature extractor,
    # and replacing the final classifier for our selected traffic sign classes.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


model = build_model(num_classes=len(SELECTED_CLASSES)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


def train_one_epoch(model, loader, criterion, optimizer, device):
    # we Train the model for one epoch on the combined real + synthetic training set.
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def evaluate(model, loader, criterion, device):
    #   then we   Evaluate the synthesized-data model on validation or test data and return
    # loss, accuracy, and prediction details.
    
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, running_correct / total, all_labels, all_preds


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.4f}")

model.load_state_dict(best_model_wts)

model_path = os.path.join(MODEL_DIR, "synthesized_resnet50_gtsrb_best.pth")
torch.save(model.state_dict(), model_path)
print("Saved model to:", model_path)

# =========================
# TEST EVALUATION
# =========================
test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

target_names = [CLASS_NAMES[inv_label_map[i]] for i in range(len(SELECTED_CLASSES))]
report = classification_report(y_true, y_pred, target_names=target_names)

print("\nClassification Report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "synthesized_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# =========================
# LOSS CURVE
# =========================
epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Synthesized Data Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "synthesized_loss_curve.png"))
plt.close()

# =========================
# ACCURACY CURVE
# =========================
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Synthesized Data Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "synthesized_accuracy_curve.png"))
plt.close()

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
class_labels_pretty = [CLASS_NAMES[cls_id] for cls_id in SELECTED_CLASSES]

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Synthesized Data Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_labels_pretty))
plt.xticks(tick_marks, class_labels_pretty, rotation=45, ha="right")
plt.yticks(tick_marks, class_labels_pretty)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "synthesized_confusion_matrix.png"))
plt.close()

# =========================
# SAVING CSV FILES
# =========================
pd.DataFrame(history).to_csv(
    os.path.join(OUTPUT_DIR, "synthesized_training_history.csv"),
    index=False
)

pd.DataFrame({
    "metric": ["test_loss", "test_accuracy", "best_val_accuracy"],
    "value": [test_loss, test_acc, best_val_acc]
}).to_csv(
    os.path.join(OUTPUT_DIR, "synthesized_test_metrics.csv"),
    index=False
)

print("Saved synthesized_classification_report.txt")
print("Saved synthesized_loss_curve.png")
print("Saved synthesized_accuracy_curve.png")
print("Saved synthesized_confusion_matrix.png")
print("Saved synthesized_training_history.csv")
print("Saved synthesized_test_metrics.csv")
print("\nSynthesized-data run completed successfully.")