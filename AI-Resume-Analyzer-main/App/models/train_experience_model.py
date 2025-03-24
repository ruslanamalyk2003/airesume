import os
import pickle
import pandas as pd
import re
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Step 1: Download the dataset using kagglehub
dataset_path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")

# Find the CSV file inside the downloaded folder
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(dataset_path, file)
        break

print(f"✅ Dataset downloaded and saved at: {dataset_path}")

# ✅ Step 2: Load the dataset
df = pd.read_csv(dataset_path)

# Ensure dataset has 'Resume' column
if "Resume" not in df.columns:
    raise ValueError("Dataset must contain 'Resume' column.")

# ✅ Step 3: Label experience level based on resume text
def label_experience(resume_text):
    text = resume_text.lower()
    if "intern" in text or "entry level" in text:
        return "Fresher"
    elif "year" in text and ("1 year" in text or "2 years" in text):
        return "Intermediate"
    elif "senior" in text or "5+ years" in text:
        return "Experienced"
    return "Intermediate"

df["experience_level"] = df["Resume"].apply(label_experience)

# ✅ Step 4: Clean resume text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    return text

df["Resume"] = df["Resume"].apply(clean_text)

# ✅ Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Resume"], df["experience_level"], test_size=0.2, random_state=42)

# ✅ Step 6: Train a model with TF-IDF + Random Forest
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectors, y_train)

# ✅ Step 7: Evaluate the model
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# ✅ Step 8: Save the trained model and vectorizer
with open("experience_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("experience_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("🎉 Experience model and vectorizer saved successfully!")
