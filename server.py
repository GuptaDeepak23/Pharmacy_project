from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
from PIL import Image
import io
import math
import cv2
import numpy as np

# -----------------------------
# Load .env file safely
# -----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env', encoding='utf-8')  # explicit encoding

# -----------------------------
# MongoDB connection
# -----------------------------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

# -----------------------------
# Reference colors for detection
# -----------------------------

@app.on_event("startup")
async def startup_db_check():
    try:
        # Try to get database names (simple check)
        await client.list_database_names()
        print("✅ MongoDB is connected successfully!")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")


REFERENCE_COLORS = [
    {"metal": "Lead", "ppm": "1000ppm", "rgb": [120, 50, 40]},
    {"metal": "Lead", "ppm": "10ppm", "rgb": [140, 70, 55]},
    {"metal": "Lead", "ppm": "0.1ppm", "rgb": [170, 100, 60]},
    {"metal": "Lead", "ppm": "0.01ppm", "rgb": [200, 160, 110]},
    {"metal": "Mercury", "ppm": "1000ppm", "rgb": [90, 60, 70]},
    {"metal": "Mercury", "ppm": "10ppm", "rgb": [170, 80, 60]},
    {"metal": "Mercury", "ppm": "0.1ppm", "rgb": [190, 120, 90]},
    {"metal": "Mercury", "ppm": "0.01ppm", "rgb": [210, 170, 120]},
    {"metal": "Mercury + Lead", "ppm": "High", "rgb": [170, 90, 90]},
    {"metal": "Mercury + Lead", "ppm": "Medium", "rgb": [200, 120, 120]},
    {"metal": "Mercury + Lead", "ppm": "Low", "rgb": [210, 150, 150]},
    {"metal": "Mercury + Lead", "ppm": "Very Low", "rgb": [230, 170, 160]}
]

# -----------------------------
# Pydantic Models
# -----------------------------
class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metal: str
    concentration: str
    status: str
    recommendation: str
    detected_rgb: List[int]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisResponse(BaseModel):
    metal: str
    concentration: str
    status: str
    recommendation: str
    detected_rgb: List[int]

# -----------------------------
# Utility functions
# -----------------------------
def euclidean_distance(rgb1: List[int], rgb2: List[int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

def get_status_and_recommendation(metal: str, ppm: str) -> tuple:
    if "1000ppm" in ppm or "High" in ppm:
        return "Highly Contaminated", "Do not consume. Seek professional water treatment immediately."
    elif "10ppm" in ppm or "Medium" in ppm:
        return "Contaminated", "Not safe for drinking. Use filtration or boiling before use."
    elif "0.1ppm" in ppm or "Low" in ppm:
        return "Slight Contamination Detected", "Boil or filter water before use."
    else:
        return "Safe", "Water is safe for consumption."

def detect_test_tube_and_extract_color(image):
    """
    Use OpenCV to detect test tube boundaries and extract color from liquid area
    """
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the test tube)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Focus on the bottom 90% of the test tube where liquid is
        liquid_y = int(y + h * 0.1)  # Start from 10% down
        liquid_h = int(h * 0.9)      # Take 90% of the height
        
        # Extract the liquid area
        liquid_area = opencv_image[liquid_y:liquid_y + liquid_h, x:x + w]
        
        if liquid_area.size > 0:
            # Convert back to RGB
            liquid_rgb = cv2.cvtColor(liquid_area, cv2.COLOR_BGR2RGB)
            
            # Reshape to get all pixels
            pixels = liquid_rgb.reshape(-1, 3)
            
            # Filter out very bright pixels (background/reflections)
            # and very dark pixels (shadows)
            filtered_pixels = []
            for pixel in pixels:
                r, g, b = pixel
                total = r + g + b
                # Keep pixels that look like liquid (not too bright, not too dark)
                if 150 < total < 650:
                    filtered_pixels.append(pixel)
            
            if filtered_pixels:
                # Calculate average color
                avg_r = int(np.mean([p[0] for p in filtered_pixels]))
                avg_g = int(np.mean([p[1] for p in filtered_pixels]))
                avg_b = int(np.mean([p[2] for p in filtered_pixels]))
                print(f"🎨 OpenCV Method - Extracted Color: RGB({avg_r}, {avg_g}, {avg_b})")
                return [avg_r, avg_g, avg_b]
    
    # Fallback: use the original method if test tube detection fails
    return None

# -----------------------------
# API Endpoints
# -----------------------------
@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), manual_color: str = Form(None)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check if manual color is provided
        print(f"🔍 Received manual_color parameter: {manual_color}")
        if manual_color:
            try:
                detected_rgb = json.loads(manual_color)
                print(f"🎨 Manual Color Selection - Using: RGB({detected_rgb[0]}, {detected_rgb[1]}, {detected_rgb[2]})")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Invalid manual color format: {e}. Falling back to automatic detection.")
                detected_rgb = None
        else:
            print("🔄 No manual color provided, using automatic detection")
            detected_rgb = None
        
        # If no manual color or manual color is invalid, try automatic detection
        if detected_rgb is None:
            # Try to detect test tube and extract color using OpenCV
            detected_rgb = detect_test_tube_and_extract_color(image)
        
        # If test tube detection fails, fall back to the original method
        if detected_rgb is None:
            print("⚠️  OpenCV test tube detection failed, using fallback method...")
            # Work with original image size for maximum color accuracy
            width, height = image.size
            
            # Focus on the bottom 90% of the image where the liquid typically is
            left = width // 6  # Start from ~17% from left
            top = int(height * 0.1)  # Start from 10% from top (focus on bottom 90%)
            right = int(width * 0.83)  # End at ~83% from left
            bottom = height  # Go all the way to the bottom
            
            cropped = image.crop((left, top, right, bottom))
            pixels = list(cropped.getdata())
            
            # Filter out very bright pixels (background) and very dark pixels (shadows)
            colored_pixels = [p for p in pixels if 150 < sum(p) < 650]
            
            if colored_pixels:
                avg_r = sum(p[0] for p in colored_pixels) // len(colored_pixels)
                avg_g = sum(p[1] for p in colored_pixels) // len(colored_pixels)
                avg_b = sum(p[2] for p in colored_pixels) // len(colored_pixels)
                print(f"🎨 Fallback Method - Extracted Color: RGB({avg_r}, {avg_g}, {avg_b})")
            else:
                # Last resort: use all pixels
                avg_r = sum(p[0] for p in pixels) // len(pixels)
                avg_g = sum(p[1] for p in pixels) // len(pixels)
                avg_b = sum(p[2] for p in pixels) // len(pixels)
                print(f"🎨 Fallback Method (All Pixels) - Extracted Color: RGB({avg_r}, {avg_g}, {avg_b})")
            
            detected_rgb = [avg_r, avg_g, avg_b]

        min_distance = float('inf')
        closest_match = None
        for ref in REFERENCE_COLORS:
            distance = euclidean_distance(detected_rgb, ref['rgb'])
            if distance < min_distance:
                min_distance = distance
                closest_match = ref

        status, recommendation = get_status_and_recommendation(
            closest_match['metal'],
            closest_match['ppm']
        )

        result = AnalysisResponse(
            metal=closest_match['metal'],
            concentration=closest_match['ppm'],
            status=status,
            recommendation=recommendation,
            detected_rgb=detected_rgb
        )
        
        print(f"✅ Final Analysis Result:")
        print(f"   Metal: {result.metal}")
        print(f"   Concentration: {result.concentration}")
        print(f"   Status: {result.status}")
        print(f"   Detected RGB: {result.detected_rgb}")
        print(f"   Closest Reference: {closest_match['rgb']} (distance: {min_distance:.2f})")

        analysis_record = AnalysisResult(
            metal=result.metal,
            concentration=result.concentration,
            status=result.status,
            recommendation=result.recommendation,
            detected_rgb=result.detected_rgb
        )
        doc = analysis_record.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.analysis_results.insert_one(doc)

        return result

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@api_router.get("/results", response_model=List[AnalysisResult])
async def get_results():
    try:
        results = await db.analysis_results.find({}, {"_id": 0}).sort("timestamp", -1).to_list(100)
        for result in results:
            if isinstance(result['timestamp'], str):
                result['timestamp'] = datetime.fromisoformat(result['timestamp'])
        return results
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")

@api_router.get("/")
async def root():
    return {"message": "AI-Integrated Heavy Metal Detection Kit API"}

# -----------------------------
# Include Router & Middleware
# -----------------------------
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Shutdown event
# -----------------------------
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# ASGI application variable for Uvicorn
application = app
