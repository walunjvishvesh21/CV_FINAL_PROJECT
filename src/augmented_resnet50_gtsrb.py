import os
import random
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image

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

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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


# # Augmentation experiment: training and evaluating ResNet50 
# on the same GTSRB subset using original data plus training-time augmentation.


def set_seed(seed=42):
     # we are Setting random seeds for reproducibility so training results are more consistent
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

     #    Here we try to Build a dataframe of training image paths for the selected classes
    #     with original class IDs and remapped labels.
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
                    "label": label_map[cls_id]
                })
    return pd.DataFrame(records)


def build_test_dataframe(test_dir, test_csv_path, selected_classes, label_map):
        # here we are Reading the official GTSRB test CSV, keeping only the selected classes, and creating
    #a dataframe containing test image paths and labels.
    test_csv = pd.read_csv(test_csv_path, sep=';')
    filtered = test_csv[test_csv["ClassId"].isin(selected_classes)].copy()
    filtered["filepath"] = filtered["Filename"].apply(lambda x: os.path.join(test_dir, x))
    filtered["label_original"] = filtered["ClassId"]
    filtered["label"] = filtered["ClassId"].map(label_map)
    return filtered[["filepath", "label_original", "label"]].reset_index(drop=True)


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

print("Train size:", len(train_split_df))
print("Val size:", len(val_split_df))
print("Test size:", len(test_df))


class GTSRBSubsetDataset(Dataset):
    #    This is a Custom PyTorch dataset that loads GTSRB images from a dataframe and applies transforms. 
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
          # This Returns the total number of samples in the dataset.
        return len(self.dataframe)

    def __getitem__(self, idx):
            #   This Loads one image and its label by index, then converts the image to RGB,
        #   applying transforms, and returns the processed image-label pair.
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================
# AUGMENTED TRAIN TRANSFORM
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


train_dataset = GTSRBSubsetDataset(train_split_df, transform=train_transform)
val_dataset = GTSRBSubsetDataset(val_split_df, transform=val_test_transform)
test_dataset = GTSRBSubsetDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def build_model(num_classes):

        #   This  Loads an ImageNet-pretrained ResNet50, freezes the backbone,
    #    and replaces the final fully connected layer for our 6-class classification task.
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
    #   This Runs one full training epoch and returns average training loss and accuracy.
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
        #     Here we are Evaluating the model on validation or test data and return average loss,
    #     accuracy, true labels, and predicted labels.
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

model_path = os.path.join(MODEL_DIR, "augmented_resnet50_gtsrb_best.pth")
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

with open(os.path.join(OUTPUT_DIR, "augmented_classification_report.txt"), "w", encoding="utf-8") as f:
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
plt.title("Augmented Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augmented_loss_curve.png"))
plt.close()

# =========================
# ACCURACY CURVE
# =========================
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Augmented Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augmented_accuracy_curve.png"))
plt.close()

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
class_labels_pretty = [CLASS_NAMES[cls_id] for cls_id in SELECTED_CLASSES]

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Augmented Confusion Matrix")
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
plt.savefig(os.path.join(OUTPUT_DIR, "augmented_confusion_matrix.png"))
plt.close()

# =========================
# SAVING CSV FILES
# =========================
pd.DataFrame(history).to_csv(
    os.path.join(OUTPUT_DIR, "augmented_training_history.csv"),
    index=False
)

pd.DataFrame({
    "metric": ["test_loss", "test_accuracy", "best_val_accuracy"],
    "value": [test_loss, test_acc, best_val_acc]
}).to_csv(
    os.path.join(OUTPUT_DIR, "augmented_test_metrics.csv"),
    index=False
)

print("Saved augmented_classification_report.txt")
print("Saved augmented_loss_curve.png")
print("Saved augmented_accuracy_curve.png")
print("Saved augmented_confusion_matrix.png")
print("Saved augmented_training_history.csv")
print("Saved augmented_test_metrics.csv")
print("\nAugmentation run completed successfully.")