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
We used the **German Traffic Sign Recognition Benchmark (GTSRB)** as the dataset for this project.

Selected classes:
- 00012 -- Priority road
- 00013 -- Yield
- 00014 -- Stop
- 00017 -- No entry
- 00025 -- Road work
- 00038 -- Keep right

The full dataset is not included in this repository due to size. To run the full experiments, download the official GTSRB dataset separately and place it in the following structure:

```text
data/GTSRB/
|-- Final_Training/
│   |___ Images/
|---Final_Test/
│   |___ Images/
|--- GT-final_test.csv

```

## Included Test Data

A small test_data/ folder is included in this repository for demonstration and reproducibility purposes. It contains a small sample of test images from the selected 6-class subset along with a labels.csv file.


## Backbone Model

The project uses ResNet50 pretrained on ImageNet as the transfer learning backbone.

## Experiment Configurations

**1. Baseline**

Trained ResNet50 on the original selected GTSRB subset with no augmentation or synthesis.

**2. Original Data + Augmentation**

Trained ResNet50 using moderate training-time augmentations such as rotation, translation, and color jitter.

**3. Original Data + Synthesized Data**

Generated synthetic training images from the original training set using perturbations such as blur, brightness changes, noise, and occlusion, then train on both real and synthetic data.

**4. Original Data + Synthesized Data + Augmentation**

Trained on the combined real and synthetic training set while also applying training-time augmentation.

## Final Results

Tested Accuracy by Configuration

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



## Robustness Accuracy Comparison

```text
Corruption	                          Baseline	                Best Model (Synthesized)
  Clean	                               93.30%	                        95.83%
  Noise	                               58.13%	                        76.98%
  Blur	                               64.17%	                        83.27%
  Occlusion	                           83.58%	                        91.43%
  Low Light	                           90.25%	                        93.68%

```

## Key Findings

Augmentation improved performance over the baseline.

Synthesized data produced the best overall performance.

The synthesized-data model outperformed the baseline under all tested corruptions.

The largest robustness gains appeared under noise and blur.



## Repository Structure
```text 

CV_FINAL_PROJECT/
|-- src/
|-- outputs/
|-- test_data/
│   |-- images/
│   |__ labels.csv
|--- data/
│   |--GTSRB/              # local only, not included in repo
|--- README.md
|--- .gitignore
|--- CV_FINAL_PROPOSAL.pdf

```

## Main Python Files

**baseline_resnet50_gtsrb.py**

Trains the baseline ResNet50 model on original data only.

**augmented_resnet50_gtsrb.py**

Trains ResNet50 with training-time augmentation.

**synthesized_resnet50_gtsrb.py**

Generates synthetic training images and trains ResNet50 on real + synthetic data.

**synth_aug_resnet50_gtsrb.py**

Trains ResNet50 on real + synthetic data with augmentation.

**robustness_test_baseline_vs_best.py**

Evaluates the baseline and best model under multiple corruptions.

**evaluate_saved_baseline.py**

Reloads the saved baseline model to recover evaluation results.

**create_small_test_data.py**

Creates the small GitHub-friendly test data folder.

**test_setup.py**

Verifies that the required Python packages are installed correctly.




## How to Run

**1. Install dependencies**

Use Python 3.12 and install the required packages:

pip install torch torchvision torchaudio pandas scikit-learn matplotlib pillow numpy

**2. Run a training configuration**

Example:

***python src/baseline_resnet50_gtsrb.py***

Other configurations:

***python src/augmented_resnet50_gtsrb.py***

***python src/synthesized_resnet50_gtsrb.py***

***python src/synth_aug_resnet50_gtsrb.py***


**3. Run robustness testing**

***python src/robustness_test_baseline_vs_best.py***

## Outputs

The outputs/ folder contains:

loss curves
accuracy curves
confusion matrices
classification reports
metrics CSV files
robustness comparison results


## Notes

Full dataset files are not included in the repository due to size.

Model weight files are also excluded from the repository if large.

The included test_data/ folder is only for demonstration and reproducibility.



## Proposal

The original project proposal is included in this repository as:

CV_FINAL_PROPOSAL.pdf