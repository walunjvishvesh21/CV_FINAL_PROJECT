# Traffic Sign Classification with Transfer Learning

## Project Title
Traffic Sign Classification with Transfer Learning: Studying the Effect of Augmentation and Synthetic Data

## Team
- Vishvesh Walunj
- Mohneet Sandhu

## Project Overview
This project studies how different data strategies affect image classification performance when fine-tuning an ImageNet-pretrained deep learning model. We use a 6-class subset of the German Traffic Sign Recognition Benchmark (GTSRB) and fine-tune a pretrained ResNet50 model under four configurations:

1. Baseline
2. Original data + augmentation
3. Original data + synthesized data
4. Original data + synthesized data + augmentation

The goal is to compare performance across these configurations and evaluate robustness under difficult image conditions such as noise, blur, occlusion, and low-light settings.

## Dataset
We use the **German Traffic Sign Recognition Benchmark (GTSRB)** as the dataset for this project.

Selected classes:
- 00012 — Priority road
- 00013 — Yield
- 00014 — Stop
- 00017 — No entry
- 00025 — Road work
- 00038 — Keep right

The full dataset is not included in this repository due to size. To run the full experiments, download the official GTSRB dataset separately and place it in the following structure:

```text
data/GTSRB/
├── Final_Training/
│   └── Images/
├── Final_Test/
│   └── Images/
└── GT-final_test.csv

```

## Included Test Data

A small test_data/ folder is included in this repository for demonstration and reproducibility purposes. It contains a small sample of test images from the selected 6-class subset along with a labels.csv file.


## Backbone Model

The project uses ResNet50 pretrained on ImageNet as the transfer learning backbone.

## Experiment Configurations

**1. Baseline**

Train ResNet50 on the original selected GTSRB subset with no augmentation or synthesis.

**2. Original Data + Augmentation**

Train ResNet50 using moderate training-time augmentations such as rotation, translation, and color jitter.

**3. Original Data + Synthesized Data**

Generate synthetic training images from the original training set using perturbations such as blur, brightness changes, noise, and occlusion, then train on both real and synthetic data.

**4. Original Data + Synthesized Data + Augmentation**

Train on the combined real and synthetic training set while also applying training-time augmentation.

## Final Results

Test Accuracy by Configuration

Baseline: **93.30%**
Augmented: **94.14%**
Synthesized: **95.83%**
Synthesized + Augmented: **95.64%**


## Best Model

The best-performing model was the Synthesized Data configuration with a test accuracy of 95.83%.

## Robustness Testing

We compared the baseline model and the best model under the following corruptions:

Clean
Noise
Blur
Occlusion
Low light