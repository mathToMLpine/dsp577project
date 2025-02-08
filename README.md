# dsp577project
This is the final project for DSP 577 

## **Contributors**  
- **Leonardo Pinheiro**  
- **Sedat Touray**  
- **Mary Klimasewiski**  
---

# **Pneumonia Detection: Performance and Resource Trade-Off Analysis**  

## **Project Overview**  
This project examines the trade-offs between computational efficiency and model performance in medical imaging. Using the **RSNA Pneumonia Detection 2018** dataset, we compare two classification models:  

1. **Feed-Forward Neural Network (FFNN) - TensorFlow**  
2. **Support Vector Machine (SVM) - Scikit-learn**  

### **Key Research Questions:**  
- Does deep learning always outperform traditional machine learning?  
- When should computational efficiency guide model selection?  

By comparing these models, we provide practical insights for selecting machine learning approaches in healthcare applications.  

## **Dataset**  
We use a subset of the **RSNA Pneumonia Detection Challenge 2018** dataset, containing **5,840 frontal chest X-rays**.  

### **Dataset Breakdown:**  
- **1,575** normal chest X-ray images  
- **4,265** pneumonia-positive X-rays (imbalanced dataset)  

### **Key Features:**  
| Feature  | Description |
|----------|------------|
| `patientID` | Unique identifier for each X-ray image |
| `X, Y, Width, Height` | Bounding box coordinates |
| `Target` | Binary classification: 1 = pneumonia, 0 = normal |

## **Methodology**  
### **Data Preprocessing:**  
- **Class Imbalance Handling:** Oversampling and undersampling.  
- **Image Processing:** Resizing and normalizing images.  
- **Data Splitting:** Training, validation, and test sets.  

### **Model Implementation:**  
#### **1. Feed-Forward Neural Network (FFNN)**  
- Multi-layer architecture optimized for pattern recognition.  
- Hyperparameter tuning to improve accuracy.  
- Implemented with TensorFlow/Keras.  

#### **2. Support Vector Machine (SVM)**  
- Kernel-based classification, efficient for smaller datasets.  
- Hyperparameter tuning (kernel type, regularization parameter).  
- Implemented with Scikit-learn.  

### **Feature Engineering:**  
- **Dimensionality Reduction:** PCA enhances SVM performance.  

### **Evaluation Metrics:**  
- **Performance Metrics:** Accuracy, precision, recall, F1-score.  
- **Computational Efficiency:** Training time, runtime, and resource utilization (CPU/GPU).  

## **Expected Outcomes**  
- **FFNN is expected to achieve higher accuracy** but demands more resources.  
- **SVM may perform comparably well** in resource-constrained settings.  
- The study offers insights into optimizing model selection for medical imaging.  

## **Project Timeline**  
| Week | Task |
|------|------|
| 3 | Data wrangling, EDA |
| 4 | Model implementation, hyperparameter tuning |
| 5 | Performance evaluation, computing metrics analysis |
| 6 | Trade-off analysis, documentation, presentation prep |
| 7 | Finalizing report and recommendations |

## **Installation & Setup**  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/mathToMLpine/dsp577project.git
   cd dsp577project
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the models:**  
   Run the FFNN and SVM models in the Jupyter Notebook:  
   [Run Models](https://github.com/mathToMLpine/dsp577project/blob/main/DSSP577_Project_with_PCA_tSNE_with_NN.ipynb)  


---  

### **Acknowledgments**  
We acknowledge the **Radiological Society of North America (RSNA)** for providing the dataset.  


