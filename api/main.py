# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow import keras 
from PIL import Image
import numpy as np
import io
import os


# FastAPI
app = FastAPI(title="Animal Face Classifier API", version="1.0")

MODEL_PATH = './best_animal_model.keras' 
IMAGE_SIZE = (128, 128)
CLASSES = ['cat', 'dog', 'wild'] 

try:
    model = keras.models.load_model('./best_animal_model.keras', compile=False) 
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None 

# (Preprocessing Function) 
def preprocess_image(img: Image.Image) -> np.ndarray:
    
    img = img.resize(IMAGE_SIZE)
    
    img_array = np.array(img, dtype=np.float32)
    
    img_array /= 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.post("/predict")
async def predict_animal(file: UploadFile = File(...)):
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check the model file path.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="The file sent is not an image.")
    
    try:
        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert('RGB') 
        

        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)
        

        predicted_class_index = np.argmax(prediction[0])
        predicted_label = CLASSES[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        

        return {
            "prediction": predicted_label,
            "confidence": f"{confidence:.4f}",
            "probabilities": {CLASSES[i]: float(prediction[0][i]) for i in range(len(CLASSES))}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal processing error occurred: {e}")


