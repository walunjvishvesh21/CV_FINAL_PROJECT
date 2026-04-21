# we created a small GitHub-friendly test_data folder by copying a few sample test 
# images per class and saving their labels in a CSV file.

import os
import shutil
import pandas as pd

ROOT_DIR = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT\data\GTSRB"
TEST_DIR = os.path.join(ROOT_DIR, "Final_Test", "Images")
TEST_CSV = os.path.join(ROOT_DIR, "GT-final_test.csv")

REPO_ROOT = r"C:\Users\Vishvesh\OneDrive\Desktop\SEM 1 PROJECTS\html\CV_FINAL_PROJECT"
SMALL_TEST_DIR = os.path.join(REPO_ROOT, "test_data")
SMALL_TEST_IMAGES_DIR = os.path.join(SMALL_TEST_DIR, "images")
SMALL_TEST_LABELS_CSV = os.path.join(SMALL_TEST_DIR, "labels.csv")

SELECTED_CLASSES = [12, 13, 14, 17, 25, 38]
IMAGES_PER_CLASS = 3

CLASS_NAMES = {
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    17: "No entry",
    25: "Road work",
    38: "Keep right"
}

os.makedirs(SMALL_TEST_IMAGES_DIR, exist_ok=True)

# Reading full test CSV
test_df = pd.read_csv(TEST_CSV, sep=';')

# we only Keep only the selected classes which is 6 .
filtered_df = test_df[test_df["ClassId"].isin(SELECTED_CLASSES)].copy()

# now we Pick small sample from each class
sampled_rows = []

for cls_id in SELECTED_CLASSES:
    cls_df = filtered_df[filtered_df["ClassId"] == cls_id].head(IMAGES_PER_CLASS)
    sampled_rows.append(cls_df)

small_df = pd.concat(sampled_rows).reset_index(drop=True)

records = []

for _, row in small_df.iterrows():
    filename = row["Filename"]
    class_id = int(row["ClassId"])

    src_path = os.path.join(TEST_DIR, filename)
    dst_filename = f"class_{class_id}_{filename}"
    dst_path = os.path.join(SMALL_TEST_IMAGES_DIR, dst_filename)

    shutil.copy2(src_path, dst_path)

    records.append({
        "image_name": dst_filename,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id]
    })

labels_df = pd.DataFrame(records)
labels_df.to_csv(SMALL_TEST_LABELS_CSV, index=False)

print("Created small test data folder at:", SMALL_TEST_DIR)
print("Total images copied:", len(records))
print("Saved labels file:", SMALL_TEST_LABELS_CSV)

# # Saved the copied image names and labels into test_data/labels.csv for GitHub reproducibility.