# Fashion-MNIST-Image-Classification-using-Transfer-Learning-with-ResNet50
1. Introduction
This project applies transfer learning using a pre-trained ResNet50 model to classify images from the Fashion MNIST dataset.
Fashion MNIST contains 28×28 grayscale images of clothing items. While traditional CNNs perform well on this dataset, transfer learning enables deeper feature extraction and improved performance.
To adapt ResNet50 for this task:
Grayscale images were converted to RGB
Images were resized to 224×224
Data augmentation was applied
The final layers of ResNet50 were fine-tuned
The resulting model achieved a validation accuracy of 91.75%, outperforming standard CNN baselines.
A Dash web application was also developed for real-time prediction and confidence visualization.
2. Dataset Overview
The Fashion MNIST dataset contains:
70,000 images (60,000 training / 10,000 testing)
10 clothing categories
Grayscale (1-channel) images of size 28×28
Before training, images are:
Normalized
Converted to 3-channel RGB
Resized to 224×224 for ResNet50 compatibility
3. Methodology
3.1 Data Preprocessing
Normalize pixel values to 
0
,
1
0,1
Convert grayscale → RGB by stacking the single channel
Resize all images to 224×224
Apply augmentation:
Random flips
Random rotation
Random zoom
3.2 Model Architecture
Base model: ResNet50 (ImageNet weights, top removed)
Added layers:
Global Average Pooling
Dense(256, ReLU)
Dense(128, ReLU)
Dense(10, Softmax)
3.3 Training Setup
Optimizer: Adam
Learning rate: 0.001 with exponential decay
Batch size: 16
Epochs: 10
Fine-tuning: last ~10 layers unfrozen
4. Results
4.1 Accuracy
The fine-tuned model achieved:
Validation Accuracy: 91.75%
4.2 Confusion Matrix
(A full matrix is included in the notebook.)
4.3 Training Curves
Plots included in the notebook:
Training vs. Validation Accuracy
Training vs. Validation Loss
4.4 Dash Application
The Dash app provides:
Image upload interface
Real-time classification
Confidence score bar charts
Interactive visualization for model understanding
Screenshots are included in the notebook.
5. Contributions
This project contributes:
Demonstration of transfer learning outperforming standard CNNs on Fashion MNIST
Full workflow inside one notebook: preprocessing → training → evaluation → deployment
A user-friendly Dash app for real-time inference
Adaptation of ResNet50 to grayscale datasets (a common real-world challenge)
6. Future Work
Integrate Grad-CAM for visual interpretability
Convert model to TensorFlow Lite or ONNX for faster real-time inference
Explore alternative architectures:
EfficientNet
MobileNet
Vision Transformers
Add hyperparameter tuning
Add model checkpointing + early stopping
Expand interactive UI in Dash
7. How to Run the Notebook
Install dependencies
pip install tensorflow keras numpy matplotlib seaborn dash plotly
Launch Jupyter
jupyter notebook
Open the notebook
fashion_mnist_resnet50.ipynb
