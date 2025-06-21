import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar dataset
df = pd.read_csv("training_dataset.csv")

# Preprocesamiento: tomar solo la primera categoría y formato para clasificación simple
df["category_main"] = df["categories"].apply(lambda x: x.split(",")[0].strip())
df["format_main"] = df["formats"].apply(lambda x: x.split(",")[0].strip())

# Preparar codificación robusta (con fallback "unknown")
text_cols = ["module_title", "roadmap_title", "type_assessment"]
label_encoders = {}

for col in text_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("unknown")
    le.fit(list(df[col].unique()) + ["unknown"])
    df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
    df[col] = le.transform(df[col])
    label_encoders[col] = le

# Features e inputs
X = df[[
    "module_title", 
    "roadmap_title", 
    "roadmap_level", 
    "avg_score", 
    "type_assessment", 
    "total_modules_done"
]]

# Codificar salidas (una categoría y un formato)
le_cat = LabelEncoder()
le_fmt = LabelEncoder()
y_cat = le_cat.fit_transform(df["category_main"])
y_fmt = le_fmt.fit_transform(df["format_main"])

# Separar entrenamiento y prueba
X_train, X_test, y_cat_train, y_cat_test, y_fmt_train, y_fmt_test = train_test_split(
    X, y_cat, y_fmt, test_size=0.2, random_state=42
)

# Entrenar dos modelos separados
model_cat = RandomForestClassifier(n_estimators=100, random_state=42)
model_fmt = RandomForestClassifier(n_estimators=100, random_state=42)

model_cat.fit(X_train, y_cat_train)
model_fmt.fit(X_train, y_fmt_train)

# Guardar modelos y encoders
joblib.dump(model_cat, "category_model.pkl")
joblib.dump(model_fmt, "format_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_cat, "category_encoder.pkl")
joblib.dump(le_fmt, "format_encoder.pkl")