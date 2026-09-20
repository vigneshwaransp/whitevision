
https://github.com/user-attachments/assets/90f23c99-8682-4173-ac4d-22b93a6adebf
# Construction Vehicle Classification using EfficientNetB0 [![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/) [![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/) [![Keras 3.15](https://img.shields.io/badge/Keras-3.15-red.svg)](https://keras.io/) [![EfficientNetB0](https://img.shields.io/badge/Architecture-EfficientNetB0-success.svg)](https://arxiv.org/abs/1905.11946) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) An end-to-end, production-grade Computer Vision system designed to classify heavy construction-site vehicles into eight canonical categories using **EfficientNetB0** transfer learning, multi-scale Test-Time Augmentation (TTA), and an Out-of-Distribution (OOD) Domain Gatekeeper on the **ConstructionXC7C** dataset. --- ## Overview Modern construction sites require real-time tracking of heavy equipment for safety monitoring, fleet allocation, maintenance scheduling, and zone access control. This project provides a complete machine learning pipeline that transforms raw object detection annotations into an optimized vehicle image classification engine, backed by interactive Streamlit and FastAPI applications. --- ## Key Features - **Multi-Source Ingestion**: Automated download and discovery supporting Hugging Face (neogpx/constructionxc7c), Kaggle CLI, and local archives. - **Universal Annotation Parsing**: Flexible parsing for Pascal VOC XML formats. - **Robust Bounding Box Cropping**: Coordinate validation, boundary clipping, configurable padding, and min-size thresholding (16px). - **Data Quality & Hygiene Audit**: Automated detection of corrupt files, zero-dimension boxes, and duplicates logged to reports/data_quality_report.csv. - **Zero-Leakage Splitting**: Grouped partitioning based on source_image_id (70.4% train, 14.8% val, 14.8% test) with mathematical overlap assertions. - **Two-Stage Transfer Learning**: - *Stage 1*: Frozen backbone feature extraction with balanced class weights. - *Stage 2*: Fine-tuning top convolutional blocks with learning rate decay. - **Out-of-Distribution Domain Gatekeeper**: Intercepts and rejects passenger cars, sports cars, artwork, pets, and everyday non-construction objects before reaching the classifier. - **Multi-Scale Test-Time Augmentation (TTA)**: Aspect-preserving letterboxing, center zoom crops (1.15x), and horizontal reflection ensuring partial vehicle crops exceed 50% confidence. - **Dual User Interfaces**: - **Streamlit Dashboard**: Dark industrial UI (#080c14), telemetry chips, equipment dossiers, zero emojis. - **FastAPI REST Service**: Production endpoints (POST /predict, GET /health) with OpenAPI Swagger documentation. - **Comprehensive Verification**: 39 automated unit tests verifying annotations, leakage, model architecture, thresholding, and domain rejection. --- ## Vehicle Classes The system detects and classifies eight canonical heavy machinery types: | ID | Class Key | Display Name | Code | Color Code | Description | |:--:|:----------|:-------------|:----:|:----------:|:------------| | 0 | bulldozer | Bulldozer | [BDZ] | #E67E22 | Crawler tractor with large front pusher blade | | 1 | dump_truck | Dump Truck | [DTK] | #F39C12 | Heavy open-bed haul truck for loose aggregates | | 2 | excavator | Excavator | [EXC] | #E74C3C | Hydraulic boom, dipper, and cab on 360-degree rotating turret | | 3 | grader | Grader | [GRD] | #3498DB | Long-wheelbase machine with center grading blade | | 4 | loader | Loader | [LDR] | #2ECC71 | Front-mounted wide bucket for scooping and loading | | 5 | mixer_truck | Mixer Truck | [MIX] | #9B59B6 | Revolving drum transit mixer for wet concrete | | 6 | mobile_crane | Mobile Crane | [CRN] | #1ABC9C | Telescopic / lattice boom crane on wheeled carrier chassis | | 7 | roller | Roller | [ROL] | #34495E | Heavy compaction roller for soil, gravel, and asphalt | Class normalization is configurable in config/classes.yaml. --- ## Architecture
Input Image (224 x 224 x 3)
       |
Aspect-Preserving Letterbox Preprocessing
       |
Domain Gatekeeper (OOD ImageNet Semantic & Centroid Filter)
       |
EfficientNetB0 Backbone (Pretrained on ImageNet)
       |
Global Average Pooling 2D (1280 features)
       |
Batch Normalization + Dropout (Rate: 0.3)
       |
Dense Layer (8 Units, Softmax Activation)
       |
Class Probabilities: [p0, p1, ..., p7], sum(p) = 1.0
--- ## Installation & Setup ### 1. Clone & Set Up Environment
bash
# Clone the repository
git clone https://github.com/vigneshwaransp/whitevision.git
cd whitevision

# Create Python virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
--- ## Inference ### Command-Line Single Image Prediction
bash
# Standard prediction with Top-3 candidates
python predict.py --image path/to/sample.jpg

# Custom Top-K and confidence threshold
python predict.py --image path/to/sample.jpg --top_k 5 --threshold 0.60

# Raw JSON output for pipelines
python predict.py --image path/to/sample.jpg --json
**Example Output (Valid Construction Equipment):**
============================================================
Prediction : Excavator [EXC]
Confidence : 94.63%
Status     : High Confidence
============================================================

Top 3 Predictions:
  [1] [EXC] Excavator      : 94.63%
  [2] [DTK] Dump Truck     : 3.56%
  [3] [LDR] Loader         : 0.85%

All Class Probabilities:
  excavator      :  94.63%  |#######################  |
  dump_truck     :   3.56%  |                         |
  loader         :   0.85%  |                         |
  roller         :   0.26%  |                         |
  mixer_truck    :   0.21%  |                         |
  bulldozer      :   0.16%  |                         |
  grader         :   0.16%  |                         |
  mobile_crane   :   0.16%  |                         |
============================================================
**Example Output (Unsupported Out-of-Distribution Image):**
============================================================
[ERROR] UNSUPPORTED IMAGE DETECTED
============================================================
Status     : Rejected: Not a Construction Vehicle
Detected   : Passenger Vehicle / Sports Car
Message    : Unsupported image: A sports car was detected (57.3% confidence).
             This system only classifies heavy construction machinery:
             Bulldozer, Dump Truck, Excavator, Grader, Loader, Mixer Truck,
             Mobile Crane, Roller.
------------------------------------------------------------
Notice     : Supported equipment:
             [BDZ] Bulldozer       [DTK] Dump Truck
             [EXC] Excavator       [GRD] Grader
             [LDR] Loader          [MIX] Mixer Truck
             [CRN] Mobile Crane    [ROL] Roller
============================================================
--- ## Applications ### 1. Streamlit Operations Console Launch the interactive industrial dashboard:
bash
streamlit run app/streamlit_app.py --server.port 8501
- Available at: http://localhost:8501 ### 2. REST API (FastAPI) Launch the high-throughput inference API server:
bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
- **Interactive Swagger Docs**: http://localhost:8000/docs - **Health Check**: GET /health - **Inference**: POST /predict (multipart/form-data with image file) --- ## Running Automated Tests Execute the automated unit test suite with pytest:
bash
pytest tests/ -v
--- ## Documentation - [**TECH_STACK.md**](TECH_STACK.md): Full technical breakdown of all frameworks, libraries, and design patterns. - [**DEPLOYMENT.md**](DEPLOYMENT.md): Production deployment guide covering Docker, Streamlit Community Cloud, Hugging Face Spaces, and Render. --- ## License This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
