import os
import pickle
import pandas as pd
import re
import kagglehub
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge  # Regression model for scoring
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ✅ Step 1: Download the dataset
dataset_path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(dataset_path, file)
        break

print(f"✅ Dataset downloaded at: {dataset_path}")

# ✅ Step 2: Load the dataset
df = pd.read_csv(dataset_path)
if "Resume" not in df.columns:
    raise ValueError("Dataset must contain a 'Resume' column.")

# ✅ Step 3: Generate Resume Scores (Manually Label or Use Rules)
# Here, we simulate scoring based on section presence
def generate_score(resume_text):
    sections = {
        "objective": 6, "summary": 6, "education": 12, "experience": 16, "internship": 6,
        "skills": 7, "hobbies": 4, "interests": 5, "achievements": 13, "certifications": 12, "projects": 19
    }
    score = sum(weight for section, weight in sections.items() if section in resume_text.lower())
    return score

df["resume_score"] = df["Resume"].apply(generate_score)

# ✅ Step 4: Convert Resume Text into TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Resume"])
y = df["resume_score"]

# ✅ Step 5: Train the Resume Scoring Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge()  # Regression model to predict scores
model.fit(X_train, y_train)

# ✅ Step 6: Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model MAE: {mae:.2f}")

# ✅ Step 7: Save the Model and Vectorizer
with open("resume_scorer.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("resume_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("🎉 Resume scoring model saved successfully!")
