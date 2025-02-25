import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# 🎉 Welcome Message
st.title("🚀 Credit Card Fraud Detection App")
st.write("🔍 **Detect if a transaction is legitimate or fraudulent using machine learning!**")

# 🗂️ Load and Process the Dataset
st.write("📊 **Loading the dataset...**")
data_path = r"C:\Users\aresa\Downloads\archive (5)\creditcard.csv"
data = pd.read_csv(data_path)
data.drop(columns='Status',inplace = True)
st.write("✅ **Dataset loaded successfully!**")

# ⚖️ Balance the Dataset
st.write("⚖️ **Balancing the dataset for fairness...**")
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)
st.write(f"📏 Legitimate transactions: {len(legit_sample)}, Fraudulent transactions: {len(fraud)}")

# 🧠 Train the Logistic Regression Model
st.write("🛠️ **Training the model...**")
X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.write(f"🎯 **Model Training Accuracy:** {train_acc * 100:.2f}%")
st.write(f"🎯 **Model Testing Accuracy:** {test_acc * 100:.2f}%")

# 🎛️ User Input for Prediction
st.write("✍️ **Enter transaction details to check its legitimacy:**")
input_features = st.text_input(
    "🔢 Enter feature values separated by commas (e.g., `0.0, -1.359807, ...`)"
)


# 🎬 Predict Button
if st.button("🔮 Predict"):
    try:
        # Validate and process input
        input_values = np.array([float(x) for x in input_features.split(",")], dtype=np.float64)
        
        if len(input_values) != X.shape[1]:
            st.error(f"⚠️ Expected {X.shape[1]} features, but got {len(input_values)}. Please try again!")
        else:
            # Make a prediction
            prediction = model.predict(input_values.reshape(1, -1))
            
            # Display the result
            if prediction[0] == 0:
                st.success("✅ **Legitimate Transaction** 🛒")
            else:
                st.error("🚨 **Fraudulent Transaction** ⚠️")
        
        # Footer only appears after the prediction
        st.write("🌟 **Thank you for using the Credit Card Fraud Detection App!**")
        
    except ValueError:
        st.error("⚠️ Invalid input! Please ensure all values are numeric and separated by commas.")
