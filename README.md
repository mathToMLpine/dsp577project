# dsp577project
This is the final project for DSP 577 

# DSP577: Project

## Performance and Resource Trade-Off Analysis of Neural Networks and Support Vector Machines on Pneumonia Detection

### Project Group:
- Leonardo Pinheiro  
- Sedat Touray  
- Mary Klimasewiski  

## Project Overview
This project aims to analyze the trade-offs between computational efficiency and model performance in medical imaging applications. Using the RSNA Pneumonia Detection 2018 dataset, we will compare the performance of two classification models:
1. **Feed-Forward Neural Network (FFNN)**
2. **Support Vector Machine (SVM)**

We seek to answer two fundamental questions:
- Does a deep learning model always outperform a traditional machine learning model?
- When should computational efficiency influence model selection?

By providing a quantitative comparison of FFNN and SVM models, this study will offer practical recommendations for selecting machine learning models in healthcare data analysis.

## Dataset
We will use the **RSNA Pneumonia Detection Challenge 2018** dataset, specifically a subset containing **5,840 frontal view chest radiographs**. The dataset consists of six key features designed to classify pneumonia evidence from patient chest X-rays.

### Dataset Breakdown:
- **1,575** normal chest X-ray images
- **4,265** pneumonia-positive chest X-ray images (indicating class imbalance)

### Feature Description:
| Feature  | Description |
|----------|------------|
| patientID | Unique identifier for each X-ray image |
| X        | Upper left x-coordinate of the bounding box |
| Y        | Upper left y-coordinate of the bounding box |
| Width    | Width of the bounding box |
| Height   | Height of the bounding box |
| Target   | Binary classification: 1 = pneumonia, 0 = normal |

## Methods
### Data Preprocessing:
- **Handling Class Imbalance:** Oversampling and undersampling techniques.
- **Data Cleaning:** Removing inconsistencies and addressing missing values.
- **Image Processing:** Resizing and normalizing images for model compatibility.
- **Data Splitting:** Dividing into training, validation, and test sets.

### Model Implementation:
1. **Feed-Forward Neural Network (FFNN) - TensorFlow:**
   - Multi-layer architecture optimized for image pattern recognition.
   - Hyperparameter tuning to enhance model accuracy.
   
2. **Support Vector Machine (SVM) - Scikit-learn:**
   - Kernel-based classification suitable for smaller datasets.
   - Hyperparameter tuning (kernel type, regularization parameter) to improve performance.

### Feature Engineering:
- **Dimensionality Reduction:** Principal Component Analysis (PCA) to enhance SVM performance.

### Model Evaluation Metrics:
- **Performance Metrics:** Accuracy, precision, recall, F1-score.
- **Computational Efficiency Metrics:** Training time, runtime, and resource utilization (CPU/GPU).

## Expected Outcomes
We anticipate the following results:
- **FFNN is expected to achieve higher classification accuracy** but will be computationally expensive.
- **SVM may perform comparably well in resource-constrained environments**, making it a viable alternative.
- The study will provide insights into model selection strategies for medical imaging applications.

## Project Deliverables & Timeline
| Week | Deliverables |
|------|-------------|
| 3 | Data wrangling and exploratory data analysis (EDA) |
| 4 | Model implementation (FFNN & SVM), hyperparameter tuning |
| 5 | Performance evaluation, computing metrics, training and runtime analysis |
| 6 | Trade-off analysis, documentation, and presentation preparation |
| 7 | Finalizing project deliverables: comprehensive comparison, recommendations, and report |

## Implications
This study will contribute valuable insights to healthcare data science by offering:
- A **quantitative comparison** between deep learning and traditional machine learning approaches.
- **Guidance on model selection** for medical imaging applications, particularly in resource-limited settings.
- **Career and skill development** for team members, enhancing their portfolios with a real-world healthcare AI application.

## Conclusion
By comparing FFNN and SVM on pneumonia detection, this project will provide actionable recommendations for optimizing machine learning model performance in medical imaging. We aim to determine whether deep learning is necessary for high accuracy or if traditional machine learning methods can offer a viable alternative in resource-constrained environments.

---



