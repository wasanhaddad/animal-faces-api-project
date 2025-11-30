#  Animal Faces Classification API  
A deep learning project that classifies animal faces (Cat, Dog, Wild) using a Convolutional Neural Network (CNN) and exposes the model through a FastAPI endpoint.

---

##  Project Overview
This project builds and trains a CNN model on the **AFHQ (Animal Faces HQ)** dataset to classify animal images into three categories:

- **Cat**
- **Dog**
- **Wild**

After training the model in Google Colab, it is exported and served through a FastAPI backend, allowing users to upload an image and receive a prediction.

---
## Dataset

This project uses the AFHQ dataset, available on Kaggle:
 https://www.kaggle.com/datasets/andrewmvd/animal-faces

---
## Project Structure
```
animal_api_project/
│── api/
│   ├── main.py
│   ├── best_animal_model.keras
│── venv/
│── .gitignore
│── requirements.txt
│── README.md
```

---

## Model Information (CNN)
The CNN was trained on Google Colab using:

- `Conv2D` layers  
- `MaxPooling2D`  
- `Dropout`  
- `Adam` optimizer  
- Image size: **128 × 128 × 3**

The final trained model is saved as:  
`best_animal_model.keras`

---

## Technologies Used

### **Modeling & Training**
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib & Seaborn  
- Google Colab  

### **Backend API**
- FastAPI  
- Uvicorn  
- Pillow  
- NumPy  

### **Version Control**
- Git  
- GitHub  

---

## How to Run the API Locally

### 1. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server
```bash
uvicorn api.main:app --reload
```

### 4. Open the interactive API docs  
FastAPI automatically generates Swagger UI:

👉 http://127.0.0.1:8000/docs

---

##  API Endpoint

### **POST /predict**

**Body:**  
Upload an image file (`.jpg`, `.png`, `.jpeg`)

**Response Example:**
```json
{
  "prediction": "cat",
  "confidence": "0.9231",
  "probabilities": {
    "cat": 0.9231,
    "dog": 0.0351,
    "wild": 0.0418
  }
}
```

---

## Future Improvements
- Add a frontend UI  
- Improve model accuracy  
- Add support for more classes  
- Deploy the API publicly  
- Add logging & testing  

---

## Author

**Wasan Haddad**
