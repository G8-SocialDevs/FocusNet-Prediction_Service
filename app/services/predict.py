from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import pickle
from app.models.task import Task
import os
import json

DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
os.makedirs(DIR, exist_ok=True)

# Cargar datos desde la BD
def get_tasks(db: Session):
    tasks = db.query(Task.Title, Task.Description, Task.category).filter(Task.category != None).all()
    return pd.DataFrame(tasks, columns=["Title", "Description", "Category"])

# Preprocesar datos
def prepare_data(df: pd.DataFrame):
    df["text"] = df["Title"] + " " + df["Description"]  # Concatenar texto
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["Category"]
    return X, y, vectorizer

# Entrenar modelo
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
    
# Guardar modelo y vectorizador
def save_model(model, vectorizer):
    with open(os.path.join(DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
# Cargar modelo y vectorizador
def load_model():
    with open(os.path.join(DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Predicción
def predict(title: str, description: str, db: Session):
    try:
        model, vectorizer = load_model()
    except FileNotFoundError:
        df = get_tasks(db)
        X, y, vectorizer = prepare_data(df)
        model = train_model(X, y)
        save_model(model, vectorizer)
    
    text = title + " " + description
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]