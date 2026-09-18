# Model Card: Construction-Site Vehicle Classifier (EfficientNetB0)

## Model Overview
- **Model Name**: EfficientNetB0 Construction-Site Vehicle Classifier
- **Architecture**: EfficientNetB0 backbone (ImageNet weights) + GlobalAveragePooling2D + BatchNormalization + Dropout(0.3) + Dense(8, softmax)
- **Input Resolution**: 224 × 224 × 3 (RGB)
- **Framework**: TensorFlow 2.x / Keras 3.x
- **Target Domain**: Construction equipment telematics, safety monitoring, autonomous job-site surveillance, fleet asset tracking.

## Dataset Provenance
- **Dataset Name**: ConstructionXC7C (`neogpx/constructionxc7c`, derived from Roboflow Universe `construction-vehicle-detection-pxc7c`)
- **Format**: High-resolution RGB images with COCO-format bounding box annotations converted to vehicle crops with boundary padding.
- **Number of Classes**: 8 Canonical Vehicle Types
  1. **Bulldozer** (`bulldozer`)
  2. **Dump Truck** (`dump_truck`)
  3. **Excavator** (`excavator`)
  4. **Grader** (`grader`)
  5. **Loader** (`loader`)
  6. **Mixer Truck** (`mixer_truck`)
  7. **Mobile Crane** (`mobile_crane`)
  8. **Roller** (`roller`)
- **Data Splitting & Leakage Prevention**: Grouped strictly by `source_image_id` into 70% Training, 15% Validation, and 15% Test partitions to ensure complete disjointness.

## Training Methodology
The training protocol follows a two-stage transfer learning procedure:

### Stage 1: Feature Extraction
- **Backbone**: EfficientNetB0 frozen (`backbone.trainable = False`).
- **Classification Head**: Trainable dense layer with dropout (0.3).
- **Optimizer**: Adam ($\text{lr} = 1\times 10^{-3}$).
- **Loss Function**: Categorical Crossentropy with balanced class weighting calculated strictly from the training partition.
- **Regularization & Callbacks**: EarlyStopping (patience=5), ModelCheckpoint (best validation loss), ReduceLROnPlateau (factor=0.2, patience=2).

### Stage 2: Fine-Tuning
- **Unfreezing**: Top 30 layers of the EfficientNetB0 backbone unfrozen.
- **Batch Normalization**: Kept strictly frozen (`layer.trainable = False`) to prevent degradation of learned statistics.
- **Optimizer**: Adam ($\text{lr} = 1\times 10^{-5}$).
- **Callbacks**: EarlyStopping (patience=4), ModelCheckpoint, ReduceLROnPlateau.

### Data Augmentation (Training Split Only)
- Random Horizontal Flip (`p=0.5`)
- Random Small Rotation ($\pm 5\%$)
- Random Zoom ($\pm 10\%$)
- Random Translation ($\pm 8\%$)
- Random Contrast adjustment ($\pm 10\%$)
- Validation & Test splits strictly unaugmented (resizing + standard EfficientNet preprocessing only).

## Performance Metrics & Evaluation
- **Evaluation Set**: Untouched test set partition.
- **Metrics Tracked**:
  - Test Loss
  - Overall Accuracy
  - Macro Precision, Recall, and F1-Score
  - Weighted Precision, Recall, and F1-Score
  - Per-class confusion matrix and high-confusion pairs (e.g. Excavator vs. Loader)
- *Note: Measured metrics are dynamically recorded in `reports/classification_report.json` upon execution.*

## Intended Use
- Automated recognition and categorisation of construction machinery from drone footage, static job-site cameras, and mobile inspection apps.
- On-site equipment utilization logging, safety exclusion-zone monitoring, and access-control automation.

## Limitations & Non-Intended Use
- **Occlusion and Extreme Night Conditions**: Severe dust storms, dense night shadows, or heavy physical obstruction (>75% occlusion) can reduce classification confidence.
- **Non-Standard Attachments**: Vehicles equipped with unusual custom attachments may occasionally be confused with neighboring functional categories (e.g., backhoes with front loaders).
- **Not for Direct Actuation**: This system provides visual classification telematics and should not be used as the sole sensor for safety-critical collision avoidance or direct autonomous machine actuation without redundant physical sensors (LiDAR, radar, proximity interlocks).
