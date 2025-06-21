from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Inicializar API
app = FastAPI()

# Cargar modelos y encoders
model_cat = joblib.load("category_model.pkl")
model_fmt = joblib.load("format_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
category_encoder = joblib.load("category_encoder.pkl")
format_encoder = joblib.load("format_encoder.pkl")

# Input esperado
class ResourceInput(BaseModel):
    module_title: str
    roadmap_title: str
    roadmap_level: int
    avg_score: int
    type_assessment: str
    categories: str  # no usado en predicción pero recibido
    formats: str     # no usado en predicción pero recibido
    total_modules_done: int

# Endpoint de predicción
@app.post("/predict")
def predict(input: ResourceInput):
    # Manejar valores no vistos con "unknown"
    def encode_input(col, value):
        le = label_encoders[col]
        if value not in le.classes_:
            return le.transform(["unknown"])[0]
        return le.transform([value])[0]

    input_data = [
        encode_input("module_title", input.module_title),
        encode_input("roadmap_title", input.roadmap_title),
        input.roadmap_level,
        input.avg_score,
        encode_input("type_assessment", input.type_assessment),
        input.total_modules_done
    ]

    # Predecir
    cat_pred = model_cat.predict([input_data])[0]
    fmt_pred = model_fmt.predict([input_data])[0]

    # Decodificar
    predicted_category = category_encoder.inverse_transform([cat_pred])[0]
    predicted_format = format_encoder.inverse_transform([fmt_pred])[0]

    return {
        "predicted_category": predicted_category,
        "predicted_format": predicted_format
    }