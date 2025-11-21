# AI-Generated Image Detector

This project aims to*detect AI-generated images using deep learning techniques. With the rapid rise of generative models, distinguishing between real and AI-generated images has become an increasingly important and challenging task.  
Our detector uses a convolutional neural network (CNN) trained on real and AI-generated images to classify whether an input image is **real** or **AI-generated**.

![Ai generated image vs authentic image](https://photutorial.com/wp-content/uploads/2024/07/a-hero-image-comparing-real-and-ai-generated-portraits-side-by-side.jpg)
*Source: https://photutorial.com/ai-image-generators-transforming-stock-photo-industry/*

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
  - [Simple Model](#simple-model)
    - [Simple Model Metrics](#simple-model-metrics)
    - [Simple Model Visualization (Grad-CAM)](#visualization-grad-cam-simple)
    - [Simple Model Chameleon Benchmark](#testing-on-chameleon-dataset-simple)
  - [Transfer Learning Model](#transfer-learning-model)
    - [Transfer Model Metrics](#transfer-model-metrics)
    - [Transfer Model Visualization (Grad-CAM)](#visualization-grad-cam-trasnfer)
    - [Transfer Model Chameleon Benchmark](#testing-on-chameleon-dataset-transfer)
- [Installation](#installation)

---

## Overview
This project focuses on **binary image classification**:
- **Class 0:** Real images (human-taken photos)
- **Class 1:** AI-generated images (synthetic content)

The goal is to explore how well machine learning can differentiate between authentic and synthetic images, and to visualize model decision patterns using **Grad-CAM** heatmaps.

---

## Dataset
The project currently uses:
- **Training data:** A curated mix of real and AI-generated images.
- **Testing data:** The [**Chameleon Dataset**]([https://chameleon.ait.ethz.ch/](https://github.com/shilinyan99/AIDE)) — a challenging benchmark designed for detecting generated content.

The Chameleon dataset includes both real and AI-generated samples produced by multiple modern generative models, providing a diverse test set.

---

## Methodology
1. **Model Architecture:**  
   A simple **Convolutional Neural Network (CNN)** was implemented as the baseline detector.

2. **Training:**  
   - Images were resized and normalized.  
   - The model was trained using **binary cross-entropy loss** and **Adam optimizer**.  
   - Early stopping and validation metrics were used to prevent overfitting.

3. **Evaluation Metrics:**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  

4. **Explainability:**  
   Grad-CAM was applied to visualize which image regions influenced the model’s predictions.

5. **Benchmark on Chameleon Dataset:**  
   Each model will be evaluated on the **Chameleon dataset**, which contains images from various generative models (e.g., StyleGAN, Stable Diffusion, DALL·E). 

![Block diagram approach](figures/approach.png)

---

## Results

### Simple Model

#### Simple Model Metrics

Initial experiments with the simple CNN achieved promising accuracy on the validation set.  

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.94      | 0.75   | 0.84     | 7995    |
| AI-Gen  | 0.79      | 0.96   | 0.87     | 7995    |

Testing Accuracy: 0.856

![Simple model chameleon confusion matrix](figures/simple_confusion.png)

![Simple model training metrics](figures/simple_metrics.png)

---

#### Simple Model Visualization (Grad-CAM)
Grad-CAM was used to produce **class activation maps**, highlighting which parts of the image most strongly influenced the model’s decision.  
This helps interpret model behavior and verify that it focuses on meaningful features (e.g., texture artifacts, background inconsistencies, etc.).

![grad cam prediction explanation](figures/grad_cam.png)

#### Simple Model Chameleon Benchmark
Testing on the Chameleon dataset demonstrated that while the model generalizes somewhat to unseen generative styles, additional fine-tuning and data augmentation are needed for robustness. 

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.55      | 0.70   | 0.61     | 5000    |
| AI-Gen  | 0.58      | 0.43   | 0.49     | 5000    |

Benchmark Testing Accuracy: 0.562

![Simple model chameleon confusion matrix](figures/simple_chameleon_confusion.png)

---

### Transfer Model

#### Transfer Model Metrics

Initial experiments with the simple CNN achieved promising accuracy on the validation set.  

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.94      | 0.75   | 0.84     | 7995    |
| AI-Gen  | 0.79      | 0.96   | 0.87     | 7995    |

Testing Accuracy: 0.856

![Simple model chameleon confusion matrix](figures/transfer_training_cm.png)

![Transfer model training metrics (pre-ft)](figures/transfer_pre_ft.png)
![Transfer model training metrics (post-ft)](figures/transfer_post_ft.png)

---

#### Transfer Model Visualization (Grad-CAM)
Grad-CAM was used to produce **class activation maps**, highlighting which parts of the image most strongly influenced the model’s decision.  
This helps interpret model behavior and verify that it focuses on meaningful features (e.g., texture artifacts, background inconsistencies, etc.).

![grad cam prediction explanation](figures/grad_cam.png)

#### Transfer Model Chameleon Benchmark
Testing on the Chameleon dataset demonstrated that while the model generalizes somewhat to unseen generative styles, additional fine-tuning and data augmentation are needed for robustness. 

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.55      | 0.70   | 0.61     | 5000    |
| AI-Gen  | 0.58      | 0.43   | 0.49     | 5000    |

Benchmark Testing Accuracy: 0.562

![Simple model chameleon confusion matrix](figures/simple_chameleon_confusion.png)

## Installation
 ```bash
git clone https://github.com/ckelly27/DSC599_Capstone.git
cd DSC599_Capstone
pip install -r requirements.txt
```

Alternatively, visit this link to the Google Colab document: [Google Colab](https://colab.research.google.com/drive/1AUaz-ZqG27AqcERgQnWXXZv8bhOnhyNp#scrollTo=KdxeiS3QVYR4)

