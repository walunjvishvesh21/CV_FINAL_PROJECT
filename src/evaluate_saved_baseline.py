import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
ROOT_DIR = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\data\GTSRB"
TEST_DIR = os.path.join(ROOT_DIR, "Final_Test", "Images")
TEST_CSV = os.path.join(ROOT_DIR, "GT-final_test.csv")
MODEL_PATH = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\models\baseline_resnet50_gtsrb_best.pth"
OUTPUT_DIR = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\outputs"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TEST DATAFRAME
# =========================
test_csv = pd.read_csv(TEST_CSV, sep=';')
test_df = test_csv[test_csv["ClassId"].isin(SELECTED_CLASSES)].copy()
test_df["filepath"] = test_df["Filename"].apply(lambda x: os.path.join(TEST_DIR, x))
test_df["label_original"] = test_df["ClassId"]
test_df["label"] = test_df["ClassId"].map(label_map)
test_df = test_df[["filepath", "label_original", "label"]].reset_index(drop=True)

print("Test size:", len(test_df))

# =========================
# DATASET
# =========================
class GTSRBSubsetDataset(Dataset):
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

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

test_dataset = GTSRBSubsetDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =========================
# LOAD MODEL
# =========================
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(SELECTED_CLASSES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# EVALUATE
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nRecovered Test Accuracy: {test_acc:.4f}\n")

target_names = [CLASS_NAMES[inv_label_map[i]] for i in range(len(SELECTED_CLASSES))]
report = classification_report(all_labels, all_preds, target_names=target_names)
print("Classification Report:\n")
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "baseline_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Recovered Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Recovered Baseline Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right")
plt.yticks(tick_marks, target_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recovered_baseline_confusion_matrix.png"))
plt.show()

print("\nSaved recovered report and confusion matrix in outputs/")