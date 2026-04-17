# 📄 Comment Category Prediction using Machine Learning

---

## **1. Abstract**

With the rapid growth of online platforms, large volumes of user-generated comments are produced daily. Manually categorizing these comments is inefficient and impractical. This project focuses on building a machine learning-based system to automatically classify comments into predefined categories.

Various models including Logistic Regression, Support Vector Machine (SVM), and Naive Bayes were implemented. Text preprocessing techniques such as tokenization and TF-IDF vectorization were applied. The models were evaluated using the F1-macro score, where the best-performing model achieved a score of **0.8143**, indicating strong performance across multiple classes.

---

## **2. Introduction**

Text classification is a fundamental task in **Natural Language Processing (NLP)**. It is widely used in:

- Sentiment analysis  
- Spam detection  
- Content moderation  
- Customer feedback analysis  

### **Problem Statement**

To develop a system that can automatically predict the category of a given comment with high accuracy and balanced performance across all classes.

---

## **3. Objectives**

- To preprocess textual data effectively  
- To convert text into numerical features using TF-IDF  
- To train multiple ML models for classification  
- To perform hyperparameter tuning for optimization  
- To evaluate models using appropriate metrics  

---

## **4. Dataset Description**

- **Source:** Kaggle dataset  
- **Type:** Text dataset containing user comments and corresponding categories  

### **Features:**
- `comment_text` (input text)  
- `category` (target label)  

### **Challenges:**
- Noisy data (punctuation, stopwords)  
- Class imbalance  
- High dimensionality  

---

## **5. Methodology**

### **5.1 Data Preprocessing**

- Lowercasing text  
- Removing punctuation and special characters  
- Tokenization  
- Stopword removal  

---

### **5.2 Feature Extraction**

**TF-IDF (Term Frequency–Inverse Document Frequency)**

- Converts text into numerical vectors  
- Assigns importance to words based on frequency  

---

### **5.3 Model Building**

#### **Logistic Regression**
- Works well for linear classification  
- Efficient and interpretable  

#### **Support Vector Machine (SVM)**
- Effective in high-dimensional space  
- Uses hyperplanes for classification  

#### **Naive Bayes**
- Based on probability  
- Fast and performs well on text data  

---

### **5.4 Hyperparameter Tuning**

- Grid Search / Randomized Search  
- Cross-validation  

**Example Parameters:**
- Logistic Regression → `C`, `penalty`  
- SVM → `kernel`, `C`  
- Naive Bayes → `alpha`  

---

## **6. Evaluation Metrics**

### **F1 Score (Macro)**

Balances precision and recall and is suitable for imbalanced datasets.

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

---

## **7. Results and Analysis**

| Model                | Performance |
|---------------------|------------|
| Logistic Regression | Good       |
| SVM                 | Better     |
| Naive Bayes         | **Best**   |

- **Best F1-macro score:** 0.8143  
- Naive Bayes performed best due to its effectiveness in handling text data  

### **Observations**
- TF-IDF significantly improved performance  
- Hyperparameter tuning boosted results  
- Simpler models can outperform complex ones in NLP  

---

## **8. System Architecture**

**Workflow:**

1. Input comment  
2. Preprocessing  
3. TF-IDF vectorization  
4. Model prediction  
5. Output category  

---

## **9. Advantages**

- Fast and efficient  
- Scalable  
- Works well with real-world text  

---

## **10. Limitations**

- Difficulty handling sarcasm and context  
- Dependent on preprocessing quality  
- Limited semantic understanding  

---

## **11. Future Work**

- Deep learning models (LSTM, BERT)  
- Word embeddings (Word2Vec, GloVe)  
- Better context handling  
- Web deployment  

---

## **12. Applications**

- Social media moderation  
- Spam filtering  
- Customer feedback analysis  
- Chatbot systems  

---

## **13. Conclusion**

This project demonstrates the effectiveness of machine learning for text classification. By applying TF-IDF and training multiple models, a strong F1-macro score of **0.8143** was achieved. The results show that even simple models like Naive Bayes can perform exceptionally well with proper preprocessing and feature engineering.

---

## **14. References**

- Kaggle Dataset  
- Scikit-learn Documentation  
- NLP research papers  

---

## 🚀 Optional Add-ons

- Add model comparison graphs  
- Deploy using Flask  
- Convert to LaTeX (IEEE format)  
- Create PPT for viva  
