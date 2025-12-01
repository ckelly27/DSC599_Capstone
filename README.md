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
- [Video Demo](#demo)

---

## Overview
This project focuses on **binary image classification**:
- **Class 0:** Real images (human-taken photos)
- **Class 1:** AI-generated images (synthetic content)

The goal is to explore how well machine learning can differentiate between authentic and synthetic images, and to visualize model decision patterns using **Grad-CAM** heatmaps.

---

## Dataset
The project currently uses the CIFAKE dataset for model training. This is a dataset containing 120,000 (32x32) images:

![Sample of training data](figures/sample.png)

A benchmark dataset, called the The [**Chameleon Dataset**]([https://chameleon.ait.ethz.ch/](https://github.com/shilinyan99/AIDE)) was used to further evaluate the models. This is a challenging benchmark that contains realistic, inauthentic images that often fool humans. 

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

![Simple model training metrics](figures/simple_training_metrics.png)

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.81      | 0.86   | 0.84     | 6400    |
| AI-Gen  | 0.85      | 0.80   | 0.83     | 6400    |

Testing Accuracy: 0.83

![Simple model testing_cm](figures/simple_testing_cm.png)

---

#### Simple Model Visualization (Grad-CAM)
Grad-CAM was used to produce **class activation maps**, highlighting which parts of the image most strongly influenced the model’s decision.  
This helps interpret model behavior and verify that it focuses on meaningful features (e.g., texture artifacts, background inconsistencies, etc.).

![simple_cam](figures/simple_cam.png)

#### Simple Model Chameleon Benchmark
Testing on the Chameleon dataset demonstrated that while the model exhibits poor generalization to unseen generative styles. The results show that the simple model often mistook ai-generated images for authentic ones, achieving a recall of 0.26 for the ai-generated class. 

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.43      | 0.51   | 0.47     | 5000    |
| AI-Gen  | 0.40      | 0.32   | 0.36     | 5000    |

Benchmark Testing Accuracy: 0.42

![Simple model chameleon confusion matrix](figures/simple_benchmark_cm.png)

---

### Transfer Model

#### Transfer Model Metrics

Initial experiments with the simple CNN demonstrated excellent performance on the validation set. Achieving a high test accuracy of 0.95, both real and ai-generated imagew were classified with near perfect precision, recall, and F1-scores. 

![transfer model training](figures/transfer_testing_cm.png)

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.95      | 0.95   | 0.95     | 6400    |
| AI-Gen  | 0.95      | 0.95   | 0.95     | 6400    |

Testing Accuracy: 0.95

![Transfer testing cm](figures/transfer_testing_cm.png)

---

#### Transfer Model Visualization (Grad-CAM)
Grad-CAM was used to produce **class activation maps**, highlighting which parts of the image most strongly influenced the model’s decision.  
This helps interpret model behavior and verify that it focuses on meaningful features (e.g., texture artifacts, background inconsistencies, etc.).

![transfer grad cam prediction explanation](figures/transfer_cam.png)

#### Transfer Model Chameleon Benchmark
Testing the transfer model on the Chameleon dataset revealed that it struggles to generalize to unseen data. Although the model achieved high performance on the validation set, it failed to maintain this effectiveness when evaluated on the Chameleon benchmark. The model showed a strong bias toward predicting images as real, with a recall of 0.83 for the real class but only 0.20 for the AI-generated class. This imbalance indicates that while the model can reliably identify authentic images, it performs poorly in detecting AI-generated ones.

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Real    | 0.60      | 0.73   | 0.66     | 5000    |
| AI-Gen  | 0.65      | 0.50   | 0.57     | 5000    |

Benchmark Testing Accuracy: 0.62

![Simple model chameleon confusion matrix](figures/transfer_benchmark_cm.png)

## Installation
 ```bash
git clone https://github.com/ckelly27/DSC599_Capstone.git
cd DSC599_Capstone
pip install -r requirements.txt
```

Alternatively, visit this link to the Google Colab document: [Google Colab](https://colab.research.google.com/drive/1AUaz-ZqG27AqcERgQnWXXZv8bhOnhyNp#scrollTo=KdxeiS3QVYR4)

## Video Demo

[Watch the video](https://youtu.be/WOjE2m9pwj0)
