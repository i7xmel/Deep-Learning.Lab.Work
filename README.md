# Deep Learning Lab Work

This repository contains 9 practical programs implementing fundamental deep learning techniques, from neural network basics to advanced architectures including CNNs, LSTMs, and object detection models.

## Programs Overview

### Program 0: Hyperparameter Tuning for Fraud Detection
- Implemented MLP for credit card fraud detection using Keras Tuner
- Used RandomSearch for hyperparameter optimization (layers, units, learning rates)
- Applied data preprocessing with ColumnTransformer (StandardScaler, OneHotEncoder)
- Achieved optimized model with binary classification for fraud prediction
- Evaluated performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix

**Screenshot**

<img width="551" height="724" alt="image" src="https://github.com/user-attachments/assets/ca67d00f-afef-4e6e-8add-152fb4f40cbd" />


---

### Program 1: Perceptron Implementation for Boolean Functions
- Built custom Perceptron class from scratch using NumPy
- Implemented perceptron training algorithm with weight updates
- Tested on fundamental boolean functions: AND, OR, NAND, XOR
- Demonstrated perceptron limitations (inability to solve XOR problem)
- Visualized error reduction over training epochs for each boolean function

**Screenshot**

<img width="369" height="347" alt="image" src="https://github.com/user-attachments/assets/a94701f5-04f9-4ea1-a4a2-9692f8d31489" />
<img width="349" height="325" alt="image" src="https://github.com/user-attachments/assets/06c63d27-78bc-4e93-8011-a8e7d5b26f88" />
<img width="368" height="324" alt="image" src="https://github.com/user-attachments/assets/be2cd8d3-f2bf-4ba9-b068-a5e8a3ed31d8" />
<img width="363" height="281" alt="image" src="https://github.com/user-attachments/assets/5691148a-47e9-45aa-919c-64b4413804cf" />


---

### Program 2: Gradient Descent Optimization
- Implemented gradient descent algorithm for 1D and 2D functions
- Optimized f(x) = x² - 2x + 2 to find global minimum
- Applied gradient descent to Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
- Configured hyperparameters: learning rate, max iterations, tolerance
- Visualized optimization path for 2D function using contour plots

**Screenshot**

<img width="531" height="445" alt="image" src="https://github.com/user-attachments/assets/559c4a9a-24c5-4868-bb32-b8d82365e1fe" />


---

### Program 3: Regularization Techniques for Heart Disease Prediction
- Implemented L1, L2, and ElasticNet regularization in neural networks
- Preprocessed heart disease dataset with one-hot encoding and scaling
- Built DNN with different regularization strengths (0.1, 0.01, 0.001)
- Compared model performance across regularization types
- Generated ROC curves and confusion matrices for model evaluation

**Screenshot**

<img width="408" height="305" alt="image" src="https://github.com/user-attachments/assets/12b34609-b351-4eb0-8499-f335ad538f1a" />
<img width="435" height="299" alt="image" src="https://github.com/user-attachments/assets/7a22d09f-d57c-4db7-bd2b-ff7f95f43ddf" />
<img width="428" height="358" alt="image" src="https://github.com/user-attachments/assets/60723e51-b204-40af-9e88-44b74cfa9682" />
<img width="408" height="307" alt="image" src="https://github.com/user-attachments/assets/893f70eb-a113-4919-bd33-e8ae42bb395d" />


---

### Program 4: Dropout Regularization for Customer Churn Prediction
- Implemented three dropout strategies: uniform dropout, layer-wise dropout, Monte Carlo dropout
- Built baseline DNN architecture without dropout for comparison
- Applied increasing dropout rates across layers (0.2, 0.3, 0.4) for layer-wise regularization
- Implemented Monte Carlo dropout for uncertainty estimation
- Compared F1-scores and accuracy across different dropout approaches

**Screenshot**

<img width="582" height="316" alt="image" src="https://github.com/user-attachments/assets/a3e79347-a636-496b-b8f2-d3cff2f73377" />
<img width="596" height="565" alt="image" src="https://github.com/user-attachments/assets/18a0208c-7c53-43a2-b0b9-e818960d9cce" />



---

### Program 5: CNN for Image Classification with Data Augmentation
- Built CNN architecture with Batch Normalization layers for image classification
- Implemented comprehensive data augmentation (rotation, shifting, shearing, zooming, flipping)
- Used ImageDataGenerator for real-time data augmentation during training
- Applied dropout regularization in fully connected layers
- Visualized training progress and generated confusion matrices

**Screenshot**

<img width="319" height="450" alt="image" src="https://github.com/user-attachments/assets/bab86f1f-3ab8-4f20-83bb-3dfc0aaccd1b" />
<img width="315" height="274" alt="image" src="https://github.com/user-attachments/assets/0bd75deb-0274-4c13-882a-3d9dde1290c5" />

---

### Program 6: Transfer Learning with VGG16/VGG19 for Binary Classification
- Implemented custom CNN architecture for binary image classification
- Applied transfer learning using pre-trained VGG16 and VGG19 models
- Compared performance of custom CNN vs. VGG16 vs. VGG19
- Used frozen base models with custom classification heads
- Generated comprehensive evaluation metrics including ROC curves and confusion matrices

**Screenshot**

<img width="461" height="509" alt="image" src="https://github.com/user-attachments/assets/fe5db200-e763-4a0b-aaab-63fd86116bb8" />
<img width="338" height="184" alt="image" src="https://github.com/user-attachments/assets/8bd776b9-8cda-4956-9656-bed252523f86" />
<img width="424" height="239" alt="image" src="https://github.com/user-attachments/assets/8201717d-5e17-4f40-9cab-dbd483d00dac" />
<img width="415" height="219" alt="image" src="https://github.com/user-attachments/assets/77a13c29-a094-4f8d-8614-6a8a5c21c054" />
<img width="414" height="318" alt="image" src="https://github.com/user-attachments/assets/c91a85c5-9a09-481e-b754-a83f40110675" />
<img width="392" height="329" alt="image" src="https://github.com/user-attachments/assets/bb4df85f-391d-49b3-bb79-19d1adf99bc4" />
<img width="400" height="316" alt="image" src="https://github.com/user-attachments/assets/35538d22-ce4e-4e25-806f-b1fcd982a78a" />


---

### Program 7: YOLOv8 for Vehicle Detection and Classification
- Implemented YOLOv8 object detection model for 21 vehicle classes
- Trained model on custom road vehicle dataset with 21 vehicle categories
- Used pre-trained YOLOv8x weights for transfer learning
- Applied data augmentation and optimization during training
- Visualized training metrics (loss curves, precision, recall, mAP) and detection results

**Screenshot**

<img width="656" height="633" alt="image" src="https://github.com/user-attachments/assets/1823599c-e504-4322-afc8-e30f14706c19" />
<img width="447" height="251" alt="image" src="https://github.com/user-attachments/assets/6cf83524-e1b0-4a99-b4a0-39e7ecc528e0" />
<img width="505" height="437" alt="image" src="https://github.com/user-attachments/assets/b8677721-2b68-4e57-91ae-a3849cafe692" />
<img width="488" height="483" alt="image" src="https://github.com/user-attachments/assets/3e5924ef-d0ce-4516-80b0-4c14c6f582b3" />


---

### Program 8: LSTM for Time Series Forecasting
- Implemented LSTM network for airline passenger forecasting
- Preprocessed time series data using MinMaxScaler normalization
- Created sequence datasets with look-back windows for time series prediction
- Built LSTM model with 4 units for sequence learning
- Evaluated forecasting performance using RMSE and visualized predictions vs. actual values

**Screenshot**

<img width="466" height="276" alt="image" src="https://github.com/user-attachments/assets/0fd165d5-f6fd-4818-8713-1e8a7f297a35" />


