# MD-Anderson-Cancer-Institute-Cancer-Diagnosis-using-ANNs


🧠 Cancer Diagnosis Using Artificial Neural Networks (ANNs)
This project demonstrates how Artificial Neural Networks (ANNs) can be used to accurately diagnose cancer using radiological data. The implementation uses TensorFlow/Keras to build and evaluate a deep learning model, simulating a deployment for the MD Anderson Cancer Institute.

📁 Dataset
Name: Breast Cancer Wisconsin (Diagnostic) Dataset

Source: Kaggle

Features: 30 numerical features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

Target: diagnosis — M (Malignant) or B (Benign)


🛠️ Project Structure

MD-Anderson-Cancer-Diagnosis/
│
├── data/
│   └── breast_cancer_data.csv
├── models/
│   └── ann_model.h5
├── notebook/
│   └── cancer_diagnosis_ann.ipynb
├── README.md
├── requirements.txt
└── main.py


🧠 Model Architecture
Input Layer: 30 neurons (one for each feature)

Hidden Layer 1: 64 neurons, ReLU activation

Dropout: 30% (to prevent overfitting)

Hidden Layer 2: 32 neurons, ReLU activation

Output Layer: 1 neuron, Sigmoid activation


📈 Training & Evaluation
Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy, Precision, Recall, F1-score

Epochs: 50

Validation Split: 20%



🔍 Insights
The model performs exceptionally well, particularly in recall, making it ideal for early cancer detection.

ANN architecture was able to generalize without overfitting due to dropout and standardized input.

With further validation, this model can serve as a clinical decision support system (CDSS).

📚 References
Kaggle: Breast Cancer Dataset

TensorFlow: https://www.tensorflow.org

MD Anderson Cancer Center (assumed institution)
