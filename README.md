

---

# **Fashion-MNIST Image Classification using Transfer Learning (ResNet50)**


---

##  **Project Overview**

This project demonstrates **transfer learning** using a pre-trained **ResNet50** model to classify images from the **Fashion-MNIST** dataset.

The dataset consists of **28×28 grayscale images**, which were adapted to ResNet50’s 224×224 RGB input through:

* Converting grayscale → RGB
* Resizing all images to 224×224
* Applying data augmentation
* Fine-tuning the final ResNet50 layers

The model achieves a **validation accuracy of 91.75%**, outperforming standard CNN baselines.

A **Dash web app** is included for real-time prediction and confidence visualization.

---

##  **Dataset Overview**

Fashion-MNIST contains:

* **70,000 images**

  * 60,000 training
  * 10,000 testing
* **10 clothing categories**
* **1-channel grayscale images** of size **28×28**

### Preprocessing Steps

* Normalize pixel values to **[0, 1]**
* Convert grayscale → RGB
* Resize to **224×224** (ResNet50 input size)
* Apply augmentation: flips, rotation, zoom

---

##  **Model Architecture**

### Base Model

* **ResNet50** pretrained on ImageNet
* Top layers removed

### Added Layers

* Global Average Pooling
* Dense(256, ReLU)
* Dense(128, ReLU)
* Dense(10, Softmax)

### Training Configuration

* Optimizer: **Adam**
* Learning Rate: **0.001** (exponential decay)
* Batch Size: **16**
* Epochs: **10**
* Fine-tuning: last ~10 layers unfrozen

---

##  **Results**

###  Accuracy

* **Validation Accuracy: 91.75%**

###  Training Curves

Included in notebook:

* Training vs Validation Accuracy
* Training vs Validation Loss

###  Confusion Matrix

Full confusion matrix provided in the notebook.

###  Dash Web App

The Dash UI provides:

* Image upload
* Real-time predictions
* Confidence score bar charts
* Interactive visualization

---

##  **Contributions**

This project highlights:

* Effective use of **transfer learning** for grayscale datasets
* Full end-to-end workflow in one notebook:
  preprocessing → training → evaluation → deployment
* A practical **Dash app** for real-time inference
* Demonstration of adapting ImageNet models to non-RGB datasets

---

##  **Future Improvements**

* Integrate **Grad-CAM** for interpretability
* Export to **TensorFlow Lite** and **ONNX** for real-time mobile inference
* Explore alternative architectures: EfficientNet, MobileNet, ViT
* Add hyperparameter tuning
* Add model checkpointing + early stopping
* Expand Dash UI for richer interaction

---

##  **How to Run the Notebook**

### 1. Install dependencies

```bash
pip install tensorflow keras numpy matplotlib seaborn dash plotly
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 3. Open the project

`fashion_mnist_resnet50.ipynb`

---


