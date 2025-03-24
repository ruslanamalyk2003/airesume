import os
import pickle
import pandas as pd
import re
import kagglehub
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(dataset_path, file)
        break

# ✅ Load dataset
df = pd.read_csv(dataset_path)

# Ensure dataset has necessary columns
if "Resume" not in df.columns or "Category" not in df.columns:
    raise ValueError("Dataset must contain 'Resume' and 'Category' columns.")

# ✅ Load NLP Model (SpaCy) for Multi-language Support
nlp = spacy.load("en_core_web_sm")  # Multilingual model

# ✅ Extract skills from resume text
def extract_skills(resume_text):
    doc = nlp(resume_text.lower())
    extracted_skills = [token.text for token in doc if token.is_alpha and len(token.text) > 2]
    return " ".join(extracted_skills)

df["skills"] = df["Resume"].apply(extract_skills)

# ✅ Convert text into TF-IDF features
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, max_features=5000)
X = vectorizer.fit_transform(df["skills"])
y = df["Category"]

# ✅ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the job classification model
job_model = LogisticRegression(max_iter=2000)
job_model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = job_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save model & vectorizer
with open("job_model.pkl", "wb") as model_file:
    pickle.dump(job_model, model_file)

with open("job_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Job model and vectorizer saved successfully!")
