# Credit-Card-Fraud-Detection
Deep learning-based credit card fraud detection using Streamlit &amp; FastAPI.

# 💳 Credit Card Fraud Detection using Deep Learning
An end-to-end project to detect fraudulent credit card transactions using a Deep Learning-based MLP model, deployed with FastAPI (backend) and Streamlit (frontend). Built and customized by Trunal Rohit (Trunal26).

## 🚀 Project Overview
Credit card fraud is a major financial issue affecting millions of customers worldwide. This project uses deep learning to identify suspicious transactions based on features like Time, Amount, and PCA-transformed features (V1 to V28). The model learns from historical data and predicts the probability of fraud for any given transaction.

## 🧠 Tech Stack
| Component | Technology Used |
|------------|-----------------|
| Programming Language | Python 3.10+ |
| Deep Learning Framework | TensorFlow / Keras |
| Backend API | FastAPI |
| Frontend Interface | Streamlit |
| Data Handling | Pandas, NumPy, Scikit-learn |
| Model Storage | .h5 (Keras model), .joblib (Scaler) |
| Version Control | Git & GitHub |

## 🗂️ Project Structure
Credit-Card-Fraud-Detection/  
├── api_model_wrapper.py → FastAPI backend serving the trained model  
├── app.py → Streamlit frontend for user interaction  
├── model/  
│   ├── mlp_model.h5 → Trained neural network model  
│   └── scaler.joblib → StandardScaler object  
├── data/  
│   └── creditcard.csv → Dataset (local use only)  
├── make_json.py → Helper script to create JSON test samples  
├── sample_tx.json → Example transaction for testing  
├── requirements.txt → Dependencies  
└── .gitignore → Ignore rules for Git

## 🧩 Model Overview
The deep learning model is a Multilayer Perceptron (MLP) trained on the Kaggle Credit Card Fraud Detection dataset. It processes 30 numerical features (Time, Amount, V1–V28) and outputs a binary classification:  
1 = Fraudulent, 0 = Legitimate.

### Model Metrics
Accuracy: 99% | Precision: 0.88 | Recall: 0.90 | F1-Score: 0.86 | AUC-ROC: 0.98

## ⚙️ How to Run Locally
git clone https://github.com/Trunal26/Credit-Card-Fraud-Detection.git  
cd Credit-Card-Fraud-Detection  
python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  
python model/train_mlp.py  
uvicorn api_model_wrapper:app --reload  
streamlit run app.py  

Then visit:  
API Docs → http://127.0.0.1:8000/docs  
Streamlit UI → http://localhost:8501/

## 📊 Dataset Information
Dataset: Kaggle - Credit Card Fraud Detection (https://www.kaggle.com/mlg-ulb/creditcardfraud)  
Transactions: 284,807  
Fraudulent cases: 492  
Highly imbalanced (~0.17% fraud)  
Features: Time, Amount, and V1–V28 (PCA-transformed)

## 🧾 Future Improvements
• Add Autoencoder or LSTM for anomaly detection  
• Add real-time monitoring dashboard  
• Use SHAP/LIME for explainability  
• Deploy full stack using Docker + Cloud

## 👨‍💻 Author
Trunal Rohit  
GitHub: https://github.com/Trunal26  
📍 India

## 🪪 License
Open-sourced under the MIT License.  

⭐ If you like this project, consider giving it a star on GitHub: https://github.com/Trunal26/Credit-Card-Fraud-Detection
