# 🤖 **AKIRA - MENTAL HEALTH CHATBOT**  

<p align="center">  
  <img src="https://github.com/user-attachments/assets/f960b56d-cd2e-4720-baf7-8ffc576734e6" width="400px" height="400px" />  
  <img src="https://github.com/user-attachments/assets/59046f0c-34f1-4dc2-b994-bbdf52d9259a" width="400px" height="400px" />  
</p>  

---  

## 📌 **1- PROJECT OVERVIEW**  
This project focuses on developing a mental health chatbot using a machine learning model trained on a dataset containing 50 different questions related to mental health topics such as anxiety, depression, PTSD, stress management, and general wellness. The chatbot is designed to classify user queries into predefined intents and generate appropriate responses.  

---  

## 📂 **2- DATASET**  
The dataset used for this project was created by the author and is available on Kaggle:  
[**Mental Health Dataset**](https://www.kaggle.com/datasets/mohamedyasino/mental-health-chatbot)  

---  

## 🔄 **3- PROJECT WORKFLOW**  

### 🛠️ **1. Data Preprocessing**  
- Cleaned and tokenized text data.  
- Applied lemmatization to standardize words.  
- Encoded text data using word embeddings.  
- Split the dataset into training and validation sets.  

---  

### 📊 **4- VISUALIZATIONS**  

![Image](https://github.com/user-attachments/assets/738a9384-22f1-4c49-87f2-724528e1963d)  

- Plotted word frequency distribution to analyze commonly used terms.  
- Displayed the distribution of intent categories to ensure dataset balance.  

---  

### 🤖 **5- MODEL TRAINING**  
- Used a deep learning model with an LSTM-based architecture.  
- Implemented a sequential model with embedding layers, LSTM layers, and dense output layers.  
- Applied categorical cross-entropy loss function and Adam optimizer.  
- Trained the model with early stopping to prevent overfitting.  

---  

### 📈 **6- MODEL EVALUATION AND FINE-TUNING**  
- Evaluated the model using:  
  - **ROUGE Score** (measuring response relevance)  
  - **F1-Score & Accuracy** (to assess classification performance)  
- Performed hyperparameter tuning to optimize learning rate and dropout rate.  

---  

### 💬 **7- AKIRA CHATBOT**  
- Tested the chatbot by feeding user queries and analyzing response accuracy.  
- Conducted manual testing with sample questions to verify intent classification.  

  ![Image](https://github.com/user-attachments/assets/d83e3a84-13c7-4375-98d2-89046b2d5f1d)![Image](https://github.com/user-attachments/assets/61b74746-70c9-4585-9304-245ce49e1efd)  

---  

## 🏆 **8- RESULTS**  
- Achieved high accuracy in intent classification.  
- The chatbot successfully generated relevant responses to mental health queries.  
- Fine-tuning improved response coherence and reduced incorrect classifications.  

---  

## 🔚 **9- CONCLUSION**  
This project successfully demonstrates how AI-powered chatbots can provide structured and supportive mental health assistance. By leveraging machine learning techniques and NLP models, the chatbot effectively classifies user queries and delivers appropriate responses. Future improvements could involve expanding the dataset, integrating real-time feedback mechanisms, and implementing a more advanced transformer-based model for enhanced response quality.  

---  

## ⚙️ **10- HOW TO USE**  

### 🖥️ **Clone the Repository**  
```bash  
git clone https://github.com/MohamedAYasin/Mental-Health-Chatbot.git  
cd Mental-Health-Chatbot  
```  

---  

### 📦 **Install Dependencies**  
```bash  
pip install -r requirements.txt  
```  

---  

## 🔗 **11- IMPORTANT LINKS**  

- 📂 **Dataset:** [**Mental Health Dataset**](https://www.kaggle.com/datasets/mohamedyasino/mental-health-chatbot)  
- 📓 **The Notebook:** [**GitHub Notebook**](https://github.com/MohamedAYasin/Mental-Health-Chatbot/blob/main/Notebook/Mental_Health.ipynb)  
- 📁 **The GitHub Repository:** [**Mental Health Chatbot**](https://github.com/MohamedAYasin/Mental-Health-Chatbot.git)
- 🌐 **Streamlit:** [**Akira Mental Health Chatbot**](https://akira-mental-health-chatbot.streamlit.app/)  
- 🎥 **Demo Video:** [**Demo Video**](https://youtu.be/NnI5l-Xzo94)  

---  

## 🚀 **12- FUTURE IMPROVEMENTS**  
- One key improvement is adjusting the confidence threshold to prevent incorrect predictions.  
- Optimized memory usage by preloading the BERT model only once, reducing overhead.  

---  

## 📜 **13- LICENSE**  
This project is licensed under the **MIT License**.  

---  

## 👨‍💻 **14- AUTHOR**  
**Developed by: © 2025 [Mohamed Ahmed Yasin](https://github.com/mohamedAYasin/)**

---
