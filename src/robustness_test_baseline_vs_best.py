import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\data\GTSRB"
TEST_DIR = os.path.join(ROOT_DIR, "Final_Test", "Images")
TEST_CSV = os.path.join(ROOT_DIR, "GT-final_test.csv")

BASELINE_MODEL_PATH = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\models\baseline_resnet50_gtsrb_best.pth"
BEST_MODEL_PATH = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\models\synthesized_resnet50_gtsrb_best.pth"

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


# =========================================================
# BUILD TEST DATAFRAME
# =========================================================
test_csv = pd.read_csv(TEST_CSV, sep=';')
test_df = test_csv[test_csv["ClassId"].isin(SELECTED_CLASSES)].copy()
test_df["filepath"] = test_df["Filename"].apply(lambda x: os.path.join(TEST_DIR, x))
test_df["label_original"] = test_df["ClassId"]
test_df["label"] = test_df["ClassId"].map(label_map)
test_df = test_df[["filepath", "label_original", "label"]].reset_index(drop=True)

print("Filtered test size:", len(test_df))


# =========================================================
# CORRUPTION FUNCTIONS
# =========================================================
def add_gaussian_noise(img, sigma=20):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_blur(img, radius=2.0):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_occlusion(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    occ_w = int(w * 0.25)
    occ_h = int(h * 0.25)

    x1 = np.random.randint(0, max(1, w - occ_w))
    y1 = np.random.randint(0, max(1, h - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h

    color = tuple(np.random.randint(0, 256, size=3).tolist())
    draw.rectangle([x1, y1, x2, y2], fill=color)
    return out

def add_low_light(img, factor=0.5):
    return ImageEnhance.Brightness(img).enhance(factor)


# =========================================================
# DATASET
# =========================================================
class CorruptedGTSRBDataset(Dataset):
    def __init__(self, dataframe, corruption="clean", transform=None):
        self.dataframe = dataframe
        self.corruption = corruption
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def apply_corruption(self, image):
        if self.corruption == "clean":
            return image
        elif self.corruption == "noise":
            return add_gaussian_noise(image, sigma=20)
        elif self.corruption == "blur":
            return add_blur(image, radius=2.0)
        elif self.corruption == "occlusion":
            return add_occlusion(image)
        elif self.corruption == "low_light":
            return add_low_light(image, factor=0.5)
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption}")

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])

        image = self.apply_corruption(image)

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================================================
# MODEL LOADING
# =========================================================
def load_resnet50_model(model_path, num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


baseline_model = load_resnet50_model(BASELINE_MODEL_PATH, len(SELECTED_CLASSES))
best_model = load_resnet50_model(BEST_MODEL_PATH, len(SELECTED_CLASSES))


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm


corruptions = ["clean", "noise", "blur", "occlusion", "low_light"]
results = []

best_clean_cm = None
baseline_clean_cm = None

for corruption in corruptions:
    print(f"\nEvaluating corruption: {corruption}")

    dataset = CorruptedGTSRBDataset(test_df, corruption=corruption, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    baseline_acc, baseline_cm = evaluate_model(baseline_model, loader)
    best_acc, best_cm = evaluate_model(best_model, loader)

    results.append({
        "corruption": corruption,
        "baseline_accuracy": baseline_acc,
        "best_model_accuracy": best_acc
    })

    print(f"Baseline Accuracy ({corruption}): {baseline_acc:.4f}")
    print(f"Best Model Accuracy ({corruption}): {best_acc:.4f}")

    if corruption == "clean":
        baseline_clean_cm = baseline_cm
        best_clean_cm = best_cm


results_df = pd.DataFrame(results)
results_csv_path = os.path.join(OUTPUT_DIR, "robustness_results_baseline_vs_best.csv")
results_df.to_csv(results_csv_path, index=False)

print("\nSaved robustness results to:", results_csv_path)


# =========================================================
# PLOT ROBUSTNESS COMPARISON
# =========================================================
x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df["baseline_accuracy"], width, label="Baseline")
plt.bar(x + width/2, results_df["best_model_accuracy"], width, label="Best Model (Synthesized)")

plt.xticks(x, results_df["corruption"])
plt.ylabel("Accuracy")
plt.title("Robustness Comparison: Baseline vs Best Model")
plt.legend()
plt.tight_layout()

robustness_plot_path = os.path.join(OUTPUT_DIR, "robustness_barplot_baseline_vs_best.png")
plt.savefig(robustness_plot_path)
plt.close()

print("Saved robustness plot to:", robustness_plot_path)


# =========================================================
# SAVE CLEAN CONFUSION MATRICES
# =========================================================
def save_confusion_matrix(cm, title, save_path):
    class_labels_pretty = [CLASS_NAMES[cls_id] for cls_id in SELECTED_CLASSES]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
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
    plt.savefig(save_path)
    plt.close()


if baseline_clean_cm is not None:
    save_confusion_matrix(
        baseline_clean_cm,
        "Baseline Confusion Matrix (Clean Test Set)",
        os.path.join(OUTPUT_DIR, "baseline_clean_confusion_matrix_for_comparison.png")
    )

if best_clean_cm is not None:
    save_confusion_matrix(
        best_clean_cm,
        "Best Model Confusion Matrix (Clean Test Set)",
        os.path.join(OUTPUT_DIR, "best_clean_confusion_matrix_for_comparison.png")
    )

print("Saved clean confusion matrices for baseline and best model.")
print("\nRobustness testing completed successfully.")