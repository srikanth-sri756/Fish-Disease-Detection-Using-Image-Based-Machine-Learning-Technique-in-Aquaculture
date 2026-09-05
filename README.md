# Fish Disease Detection Using Image-Based Machine Learning Technique in Aquaculture

A Tkinter desktop application that classifies fish images as **Fresh** or **Infected**, comparing four machine-learning algorithms on a CLAHE + LAB colour-space feature pipeline.

## Overview

Disease outbreaks in aquaculture ponds are expensive and spread quickly, and manual inspection does not scale. This project applies image-based screening: fish photographs are contrast-enhanced, converted into a colour space that exposes lesion and discolouration cues, then classified.

Four classifiers — Decision Tree, Logistic Regression, Naive Bayes and SVM (the proposed approach) — are trained on the same features and scored on accuracy, precision, recall, F1-score, sensitivity and specificity, so the SVM's advantage is visible rather than asserted.

## Image preprocessing pipeline

Each image passes through the following stages before it becomes a feature vector:

1. **Bicubic interpolation** — resize to 150×150 using `INTER_CUBIC`.
2. **Greyscale conversion** — drop chroma before contrast work.
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation (`clipLimit = 5`) to bring out local texture.
4. **RGB → LAB conversion** — separates lightness from colour opponents, making discolouration more separable.
5. **Flatten & normalise** — ravel to a 1-D vector and scale to `[0, 1]`.

The processed arrays are cached to `model/X.npy` and `model/Y.npy`, so subsequent runs skip preprocessing entirely.

## Features

- **Dataset upload** — point the app at a folder containing `FreshFish/` and `InfectedFish/` subdirectories.
- **One-click preprocessing** — runs interpolation, CLAHE and LAB conversion, then performs an 80/20 train/test split and previews a sample processed image.
- **Four trainable classifiers** — Decision Tree, Logistic Regression, Naive Bayes and SVM.
- **Full metric reporting** — sensitivity, specificity, precision, recall, F1-score and accuracy per algorithm, plus a Seaborn confusion-matrix heatmap.
- **Comparison graph** — bar chart contrasting all four algorithms across every metric.
- **Single-image prediction** — load an image from `testImages/` and get a Fresh/Infected verdict rendered on the image.
- **Dataset augmentation** — `Augmentation.py` expands the infected class with rotation, shift, shear, zoom, channel-shift and flip transforms.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.7 |
| GUI | Tkinter with a custom rounded-button widget |
| Computer vision | OpenCV (CLAHE, colour-space conversion, resizing) |
| ML | scikit-learn (SVC, DecisionTree, LogisticRegression, MultinomialNB) |
| Augmentation | Keras `ImageDataGenerator` |
| Data & plots | pandas, NumPy, Matplotlib, Seaborn |

## Project structure

```
.
├── Main.py             # Tkinter app: preprocessing, training, evaluation, prediction
├── Augmentation.py     # Offline dataset augmentation for the infected class
├── CustomButton.py     # TkinterCustomButton rounded-button widget
├── run.bat             # python Main.py
├── test.py             # Scratch experiments
├── Dataset/
│   ├── FreshFish/      # Healthy fish images (label 0)
│   └── InfectedFish/   # Diseased fish images (label 1)
├── model/
│   ├── X.npy           # Cached feature matrix
│   └── Y.npy           # Cached labels
└── testImages/         # Unseen samples for the prediction step
```

## Getting started

### Prerequisites

- Python 3.7

### Installation

```bash
git clone https://github.com/srikanth-sri756/4.Fish-Disease-Detection-Using-Image-Based-Machine-Learning-Technique-in-Aquaculture.git
cd 4.Fish-Disease-Detection-Using-Image-Based-Machine-Learning-Technique-in-Aquaculture

pip install numpy pandas scikit-learn opencv-python matplotlib seaborn keras tensorflow
```

### Running

```bash
python Main.py
```

or double-click `run.bat` on Windows.

## Usage

1. **Upload Fish Dataset** — select the `Dataset` folder.
2. **Run Interpolation, CLAHE & LAB** — preprocesses every image and reports the train/test split sizes. A sample processed image opens in an OpenCV window; press any key to continue.
3. **Run Decision Tree**, **Run Logistic Regression**, **Run Naive Bayes**, **Run Propose SVM Algorithm** — train each classifier in turn and read its metrics. Close each confusion-matrix window to proceed.
4. **Comparison Graph** — view all algorithms side by side.
5. **Predict Fish Status** — load an image from `testImages/` to get a verdict.

> Run the algorithms in the listed order — the comparison graph reads from shared metric lists populated in that sequence.

## Notes

- Preprocessing produces high-dimensional flattened vectors (150 × 150 × 3), so the first training pass is memory-hungry. Delete `model/X.npy` and `model/Y.npy` to force a rebuild after changing the dataset.
- `MultinomialNB` requires non-negative features; the pipeline's `[0, 1]` scaling satisfies this.
- `Augmentation.py` writes to an `aug/` directory — create it (with a matching class subfolder) before running.

## License

Released for academic and educational use.
