# 📄 Comment Category Prediction

---

## Project Overview

This project aims to build a machine learning model to predict comment categories using both **text data** and **metadata features**.

### Problem Statement
- **Task**: Predict category of social media comments  
- **Input**: Text comments + engagement/demographic features  
- **Output**: Category label (0, 1, 2, 3)  
- **Evaluation Metric**: **F1-Macro Score** (due to class imbalance)

---

## Approach

1. Exploratory Data Analysis (EDA)  
2. Data Preprocessing & Feature Engineering  
3. Text Feature Extraction (TF-IDF)  
4. Feature Selection  
5. Model Building & Comparison  
6. Hyperparameter Tuning  
7. Final Model Training  
8. Prediction Generation  

---

## 1. Data Loading

Datasets used:
- Training dataset  
- Test dataset  
- Sample submission  

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Dataset Overview
- Checked dataset shape, structure, and data types  

---

### 2.2 Missing Values Analysis
- Identified missing values in train and test datasets  
- Ensured proper handling before modeling  

---

### 2.3 Feature Cardinality

- High cardinality columns identified:
  - `created_date` (very high unique values)
  - `comment` (text data)

**Decision:**
- Drop or transform high-cardinality features  
- Apply vectorization on text  

---

### 2.4 Feature Summary

#### Numerical Features
- Statistical summary using `.describe()`

#### Categorical Features
- Unique value counts per column  

---

### 2.5 Target Distribution Analysis

⚠️ **Critical Observation: Dataset is Highly Imbalanced**

- Class 0 → ~57% (majority)  
- Class 3 → ~2.7% (minority)  

### Design Decisions
- Use `class_weight = 'balanced'`  
- Use **F1-Macro instead of Accuracy**  

---

### 2.6 Outlier Detection
- Used IQR method  
- Handled later using transformations  

---

### 2.7 Feature Correlation

- Identified most important numerical features  
- Example:
  - `if_2` → highest correlation  
  - `upvote`, `downvote` → moderate impact  

💡 Insight:  
Text features contribute more than metadata  

---

## 3. Data Preprocessing

- Lowercasing text  
- Removing special characters  
- Handling missing values  
- Encoding categorical variables  

---

## 4. Feature Engineering

### 4.1 TF-IDF Vectorization

- Converts text into numerical representation  
- Captures importance of words  

---

### 4.2 Feature Combination

- Combined:
  - TF-IDF features (text)  
  - Numerical features  
  - Encoded categorical features  

---

## 5. Model Building

Models used:

### 5.1 Logistic Regression
- Baseline linear model  
- Works well with sparse data  

---

### 5.2 Support Vector Machine (SVM)
- Effective in high-dimensional space  
- Good for text classification  

---

### 5.3 Naive Bayes
- Probabilistic model  
- Very effective for NLP tasks  

---

## 6. Model Evaluation

### Metric Used: F1-Macro

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

### Why F1-Macro?
- Handles class imbalance  
- Treats all classes equally  

---

## 7. Results

| Model                | Performance |
|---------------------|------------|
| Logistic Regression | Good       |
| SVM                 | Better     |
| Naive Bayes         | **Best**   |

- **Best Score**: 0.8143 (F1-Macro)  

---

## 8. Key Insights

- Text features dominate prediction performance  
- Metadata features provide minor improvements  
- TF-IDF is highly effective  
- Naive Bayes outperforms due to text distribution  

---

## 9. Final Pipeline

1. Input comment  
2. Preprocessing  
3. TF-IDF transformation  
4. Feature combination  
5. Model prediction  
6. Output category  

---

## 10. Advantages

- Efficient on large datasets  
- Works well with sparse data  
- Scalable solution  

---

## 11. Limitations

- Cannot understand deep context  
- Struggles with sarcasm  
- Limited semantic understanding  

---

## 12. Future Work

- Use deep learning models (LSTM, BERT)  
- Use embeddings (Word2Vec, GloVe)  
- Improve feature engineering  
- Deploy as web application  

---

## 13. Conclusion

This project successfully builds a robust text classification system using machine learning. By combining TF-IDF features with metadata and applying multiple models, the system achieves a strong **F1-Macro score of 0.8143**.

The results highlight that **simple models like Naive Bayes can outperform complex ones when paired with proper preprocessing and feature engineering**.

---

## 14. References

- Kaggle Dataset  
- Scikit-learn Documentation  
- NLP Research Papers  

---
