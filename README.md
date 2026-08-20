# MNIST Digit Recognition

This project focuses on handwritten digit recognition using the MNIST dataset.

The project includes two different approaches:

- Comparison of several machine learning algorithms
- Handwritten digit recognition using a Convolutional Neural Network (CNN)

The project also includes an interactive interface where the user can draw a digit and receive a prediction from the trained CNN model.

## Features

- MNIST handwritten digit dataset
- Machine learning model comparison
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Model evaluation and accuracy comparison
- Confusion matrix
- Sample prediction visualization
- Interactive handwritten digit drawing
- Real-time digit prediction


## 1. Machine Learning Model Comparison

The model_comparison.py file compares several traditional machine learning algorithms.

The following models are evaluated:

Logistic Regression
Support Vector Machine (SVM)
Random Forest
Multi-Layer Perceptron (MLP)

The models are trained and tested using Scikit-learn's built-in Digits dataset.

Digits Dataset

The Scikit-learn Digits dataset contains handwritten digit images with a resolution of:

8 × 8 pixels

The dataset contains digits from:

0 1 2 3 4 5 6 7 8 9

The data is divided into training and test sets using train_test_split.

The models are then trained and their accuracy scores are compared.

The program also generates a classification report and a confusion matrix for the best-performing model.

Run the model comparison with:

python model_comparison.py

An example visualization of the dataset is saved as:

ornek_rakamlar.png

The confusion matrix is saved as:

karisiklik_matrisi.png

## 2. CNN Model

The train_cnn.py file trains a Convolutional Neural Network using the MNIST dataset.

Unlike the Scikit-learn Digits dataset used for model comparison, the MNIST dataset contains:

28 × 28 pixel

grayscale handwritten digit images.

The MNIST dataset contains handwritten digits from 0 to 9.

The dataset is automatically downloaded using Torchvision if it is not already available.

datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
CNN Architecture

The CNN model consists of:

Input
  ↓
Convolutional Layer
  ↓
ReLU
  ↓
Max Pooling
  ↓
Convolutional Layer
  ↓
ReLU
  ↓
Max Pooling
  ↓
Dropout
  ↓
Flatten
  ↓
Fully Connected Layer
  ↓
Dropout
  ↓
Output Layer

The final layer contains 10 outputs representing the digits:

0 - 9

The model uses:

2 convolutional layers
Max pooling
ReLU activation
Dropout
Fully connected layers
Adam optimizer
Cross-entropy loss

## 3. CNN Training

To train the CNN model, run:

python train_cnn.py

The program automatically downloads the MNIST dataset if necessary.

The model is trained for multiple epochs and evaluated on the MNIST test dataset after each epoch.

The program prints information such as:

Epoch 1/5
Loss: ...

Test sonucu -> Ortalama kayıp: ...
Doğruluk: .../10000 (%...)

The model with the highest test accuracy is saved as:

en_iyi_model.pth

This saved model is used later by the user prediction application.

## 4. CNN Prediction Visualization

After training, the program selects sample images from the MNIST test dataset and predicts their digits.

Correct predictions and incorrect predictions are visually distinguished.

The generated image is saved as:

cnn_tahmin_ornekleri.png

Example:

Input Image → CNN → Prediction

The visualization helps demonstrate how the trained CNN performs on unseen handwritten digits.

## 5. User Digit Prediction

The kullanici_tahmin.py file provides an interactive interface for handwritten digit recognition.

The user can draw a digit using the mouse.

The application then processes the drawing and sends it to the trained CNN model.

The prediction process is:

User Drawing
     ↓
Image Processing
     ↓
Resize to 28 × 28
     ↓
Normalization
     ↓
Trained CNN Model
     ↓
Prediction
     ↓
Confidence Score

For example:

User draws:

   7

Prediction:

Tahmin: 7
Güven: %98.32

The application uses the previously trained:

en_iyi_model.pth

model.

Therefore, the model does not need to be trained again every time the user wants to make a prediction.

## 6. Installation

Clone the repository:

git clone https://github.com/USERNAME/mnist-digit-recognition.git

Move into the project directory:

cd mnist-digit-recognition

Install the required libraries:

pip install -r requirements.txt

## 7. Usage
Step 1 - Model Comparison
Run:
python model_comparison.py

This program compares:

Logistic Regression
SVM
Random Forest
MLP

and displays their accuracy scores.

Step 2 - Train the CNN

Run:
python train_cnn.py
The MNIST dataset will be downloaded automatically if it is not already available.

After training, the best model will be saved as:
en_iyi_model.pth

Step 3 - Draw a Digit

After the CNN has been trained, run:

python kullanici_tahmin.py

A drawing window will appear.

Draw a digit using the mouse and click:

Tahmin Et

The application will display the predicted digit and its confidence score.

To draw another digit, click:

Temizle

and draw again.

## 8. Technologies

The project was developed using:

Python
PyTorch
Torchvision
Scikit-learn
NumPy
Matplotlib
Pillow
Tkinter

## 9. Datasets

Two different handwritten digit datasets are used in this project.

Scikit-learn Digits Dataset

Used for traditional machine learning model comparison.

Image size:

8 × 8 pixels

Used by:

model_comparison.py
MNIST Dataset

Used for CNN training and user digit prediction.

Image size:

28 × 28 pixels

Used by:

train_cnn.py
kullanici_tahmin.py
Dataset Difference

The two datasets are intentionally used for different parts of the project.

Machine Learning Comparison
        ↓
Scikit-learn Digits Dataset
        ↓
8 × 8 images


CNN Training
        ↓
MNIST Dataset
        ↓
28 × 28 images

## 10. Results

The project generates several visual outputs during execution.

Sample Digits

The model comparison program generates sample handwritten digits:

ornek_rakamlar.png
Confusion Matrix

The confusion matrix shows which digits are correctly and incorrectly classified:

karisiklik_matrisi.png
CNN Predictions

The CNN prediction examples are saved as:

cnn_tahmin_ornekleri.png

## 11. Requirements

The main dependencies are listed in requirements.txt:

torch
torchvision
numpy
matplotlib
scikit-learn
pillow

## 12. Notes
An internet connection is required during the first MNIST download.
The CNN model must be trained before running kullanici_tahmin.py.
The en_iyi_model.pth file is generated automatically by cnn_egitim.py.
The user prediction application expects the trained model file to be in the project directory.
The user-drawn image is converted to the 28 × 28 format expected by the CNN.

## License
This project is created for educational and learning purposes.
