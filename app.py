# Import necessary packages
from tensorflow.keras.applications.vgg16 import preprocess_input
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import uvicorn
import logging
import nest_asyncio
import os
import requests

# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your frontend setup
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dropbox direct link for the model file
model_url = 'https://www.dropbox.com/scl/fi/zohbshw4ihgvlcw3fqfhh/best_model.keras?rlkey=m5bkm3udewhtzw3lyeoilmx72&dl=1'
model_path = 'best_model.keras'

# Download the model if it doesn't already exist
if not os.path.exists(model_path):
    try:
        logger.info("Downloading model from Dropbox...")
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise HTTPException(status_code=500, detail="Failed to download model")

# Load the model
try:
    model = load_model(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Image dimensions
img_height = 244
img_width = 244

# Define class names
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Preprocessing function
def preprocess_image(image_data: bytes) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize((img_height, img_width))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Image preprocessing failed")

# Function to get recommendations based on class
def get_recommendations_and_tips(predicted_class_name):
    recommendations_tips = {
        "Bird-drop": {
            "recommendations": [
                "Regular cleaning of the solar panels is recommended.",
                "Install bird deterrent systems to reduce the chances of droppings."
            ],
            "tips": [
                "Consider setting up ultrasonic bird repellents to prevent birds from approaching.",
                "Use automated cleaning systems for frequent bird activity zones."
            ]
        },
        "Clean": {
            "recommendations": [
                "No action required. The panels are in good condition.",
                "Regular monitoring to ensure continued efficiency."
            ],
            "tips": [
                "Set a schedule for regular visual inspections even if panels look clean.",
                "Implement smart monitoring systems to track panel performance."
            ]
        },
        "Dusty": {
            "recommendations": [
                "Clean the panels to ensure optimal performance.",
                "Consider installing protective screens to reduce dust accumulation."
            ],
            "tips": [
                "Use water-efficient cleaning techniques to conserve resources while maintaining cleanliness.",
                "Install panels at a slight tilt to reduce dust buildup over time."
            ]
        },
        "Electrical-damage": {
            "recommendations": [
                "Contact a certified technician to inspect the electrical system.",
                "Do not attempt to repair electrical damage without professional assistance."
            ],
            "tips": [
                "Perform regular electrical inspections to catch minor issues early.",
                "Keep the area around electrical components clean and dry to avoid shorts."
            ]
        },
        "Physical-Damage": {
            "recommendations": [
                "Inspect the panels for cracks or breaks.",
                "Consider replacing or repairing damaged panels to avoid energy losses."
            ],
            "tips": [
                "Install protective barriers around panels to avoid future physical damage.",
                "Use high-quality tempered glass panels for increased durability."
            ]
        },
        "Snow-Covered": {
            "recommendations": [
                "Remove snow from the panels to restore energy generation.",
                "Install snow guards to prevent heavy accumulation."
            ],
            "tips": [
                "Use non-abrasive tools or warm water to clear snow without damaging the panels.",
                "Install heated cables to melt snow in colder climates."
            ]
        }
    }
    return recommendations_tips.get(predicted_class_name, {"recommendations": ["No recommendations"], "tips": ["No tips"]})

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            logger.warning(f"Unsupported file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        image_data = await file.read()
        logger.info(f"File received: {file.filename}, size: {len(image_data)} bytes")

        img_array = preprocess_image(image_data)
        predictions = model.predict(img_array)

        if predictions.shape[-1] > 1:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1)

        confidence = float(np.max(predictions))
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class]

        rec_and_tips = get_recommendations_and_tips(predicted_class_name)
        return JSONResponse(content={
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "recommendations": rec_and_tips["recommendations"],
            "tips": rec_and_tips["tips"]
        })
    except HTTPException as http_exception:
        logger.error(f"HTTP error: {http_exception.detail}")
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Main block for FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

