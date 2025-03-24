import os
import pickle
import pandas as pd
import re
import kagglehub
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ✅ Download dataset
dataset_path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(dataset_path, file)
        break

# ✅ Load dataset
df = pd.read_csv(dataset_path)

# Ensure dataset has necessary columns
if "Resume" not in df.columns:
    raise ValueError("Dataset must contain a 'Resume' column.")

# ✅ Load NLP Model (SpaCy) for Skill Extraction
nlp = spacy.load("en_core_web_sm")

# ✅ Predefined skill keywords
skill_keywords = set([
    "python", "machine learning", "deep learning", "tensorflow", "keras", "pytorch",
    "data science", "sql", "java", "c++", "html", "css", "javascript", "flask", "react",
    "django", "android", "ios", "swift", "kotlin", "ui/ux", "figma", "adobe photoshop"
])

# ✅ Extract skills from resume text
def extract_skills(resume_text):
    doc = nlp(resume_text.lower())
    extracted_skills = {token.text for token in doc if token.text in skill_keywords}
    return " ".join(extracted_skills)  # Change to space-separated string

df["skills"] = df["Resume"].apply(extract_skills)

# ✅ Use default `token_pattern` instead of custom tokenizer
vectorizer = TfidfVectorizer(lowercase=True)  # No tokenizer
X = vectorizer.fit_transform(df["skills"])

# ✅ Train KNN model for skills recommendation
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(X)

# ✅ Define course recommendations
courses = {
    "Data Science": ["Machine Learning with Python", "Deep Learning Specialization", "TensorFlow Developer"],
    "Web Development": ["Full-Stack Web Development", "React & Node.js Masterclass"],
    "Android Development": ["Android App Development with Kotlin", "Flutter for Beginners"],
    "IOS Development": ["Swift UI for iOS", "iOS Development with Objective-C"],
    "UI/UX": ["Adobe XD & Figma for Designers", "User Experience Research"],
}

# ✅ Save models & vectorizer
with open("skills_model.pkl", "wb") as model_file:
    pickle.dump(knn, model_file)

with open("skills_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)  # Now safe to pickle

with open("courses.pkl", "wb") as course_file:
    pickle.dump(courses, course_file)

print("🎉 Skills & courses model saved successfully!")
