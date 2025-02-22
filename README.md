                                  # **Mental Health Chatbot**

<p align="center">
  <img src="https://github.com/user-attachments/assets/f960b56d-cd2e-4720-baf7-8ffc576734e6" width="400px" height="400px" />
  <img src="https://github.com/user-attachments/assets/59046f0c-34f1-4dc2-b994-bbdf52d9259a" width="400px" height="400px" />
</p>

## **Project Overview**
This project focuses on developing a mental health chatbot using a machine learning model trained on a dataset containing 50 different questions related to mental health topics such as anxiety, depression, PTSD, stress management, and general wellness. The chatbot is designed to classify user queries into predefined intents and generate appropriate responses.

## **Dataset**
The dataset used for this project was created by the author and is available on Kaggle:
[**Mental Health Dataset**](https://www.kaggle.com/datasets/mohamedyasino/mental-health-chatbot)

## **Project Workflow**

### **1. Data Preprocessing**
- Cleaned and tokenized text data.
- Applied lemmatization to standardize words.
- Encoded text data using word embeddings.
- Split the dataset into training and validation sets.

### **2. Visualizations**

![Image](https://github.com/user-attachments/assets/738a9384-22f1-4c49-87f2-724528e1963d)

- Plotted word frequency distribution to analyze commonly used terms.
- Displayed the distribution of intent categories to ensure dataset balance.

### **3. Model Training**
- Used a deep learning model with an LSTM-based architecture.
- Implemented a sequential model with embedding layers, LSTM layers, and dense output layers.
- Applied categorical cross-entropy loss function and Adam optimizer.
- Trained the model with early stopping to prevent overfitting.

### **4. Model Evaluation & Fine-Tuning**
- Evaluated the model using:
  - **ROUGE Score** (measuring response relevance)
  - **F1-Score & Accuracy** (to assess classification performance)
- Performed hyperparameter tuning to optimize learning rate and dropout rate.

### **5. Testing the Chatbot**
- Tested the chatbot by feeding user queries and analyzing response accuracy.
- Conducted manual testing with sample questions to verify intent classification.

## **Results**
- Achieved high accuracy in intent classification.
- The chatbot successfully generated relevant responses to mental health queries.
- Fine-tuning improved response coherence and reduced incorrect classifications.

## **Conclusion**
This project successfully demonstrates how AI-powered chatbots can provide structured and supportive mental health assistance. By leveraging machine learning techniques and NLP models, the chatbot effectively classifies user queries and delivers appropriate responses. Future improvements could involve expanding the dataset, integrating real-time feedback mechanisms, and implementing a more advanced transformer-based model for enhanced response quality.

---
### **How to Use**
1. Clone the repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Run the chatbot script to interact with the model.

---
### **Author**
Developed by Mohamed Yasin


