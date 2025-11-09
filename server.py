from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict
import uuid
from datetime import datetime, timezone
from PIL import Image
import io
import math
import cv2
import numpy as np

# -----------------------------
# Load .env file
# -----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env', encoding='utf-8')

# -----------------------------
# Logging configuration (needed early for AI service check)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# MongoDB connection
# -----------------------------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Check if AI service should be used (default: False/disabled)
USE_AI = os.getenv('USE_AI', 'false').lower() == 'true'

# Conditionally import and initialize AI service
if USE_AI:
    from ai_service import ai_service
    ai_service_enabled = ai_service.enabled
else:
    ai_service = None
    ai_service_enabled = False
    logger.info("AI service disabled. Set USE_AI=true in .env to enable.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

# -----------------------------
# Reference colors for detection
# -----------------------------
REFERENCE_COLORS = [
    {"metal": "Copper", "ppm": "Standard", "rgb": [90, 50, 25]},
    {"metal": "Copper", "ppm": "Standard", "rgb": [132, 103, 38]},
    {"metal": "Copper", "ppm": "Standard", "rgb": [57, 31, 33]},
    {"metal": "Copper", "ppm": "Standard", "rgb": [111, 49, 47]},
    {"metal": "Copper", "ppm": "Standard", "rgb": [186, 96, 68]},
    {"metal": "Copper", "ppm": "Standard", "rgb": [199, 141, 92]}
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
    ai_recommendations: Dict = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisResponse(BaseModel):
    metal: str
    concentration: str
    status: str
    recommendation: str
    detected_rgb: List[int]
    ai_recommendations: Dict = {}

# -----------------------------
# Utility functions
# -----------------------------
def euclidean_distance(rgb1: List[int], rgb2: List[int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

# Convert RGB → HSV
def rgb_to_hsv_tuple(rgb: List[int]):
    arr = np.uint8([[rgb]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0][0]
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))

# Calculate HSV color distance (circular hue)
def hsv_distance(hsv1: tuple, hsv2: tuple) -> float:
    dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0]))
    ds = hsv1[1] - hsv2[1]
    dv = hsv1[2] - hsv2[2]
    return math.sqrt(dh * dh + ds * ds + dv * dv)

# Optional: normalize lighting
def normalize_rgb(rgb: List[int]) -> List[int]:
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0][0]
    return [int(norm[0]), int(norm[1]), int(norm[2])]

def get_status_and_recommendation(metal: str, ppm: str) -> tuple:
    if "1000ppm" in ppm or "High" in ppm:
        return "Highly Contaminated", "Do not consume. Seek professional water treatment immediately."
    elif "10ppm" in ppm or "Medium" in ppm:
        return "Contaminated", "Not safe for drinking. Use filtration or boiling before use."
    elif "0.1ppm" in ppm or "Low" in ppm:
        return "Slight Contamination Detected", "Boil or filter water before use."
    else:
        return "Safe", "Water is safe for consumption."

def get_basic_ai_recommendations(metal: str, concentration: str) -> Dict:
    """Basic recommendations when AI service is disabled"""
    if "1000ppm" in concentration or "High" in concentration:
        return {
            "immediate_actions": ["DO NOT DRINK THIS WATER", "Contact emergency services", "Use bottled water immediately"],
            "treatment_options": ["Professional water treatment required", "Whole house filtration system", "Contact certified water specialist"],
            "health_risks": "High concentration of heavy metals poses immediate health risks. Seek medical attention if exposed.",
            "prevention_tips": ["Stop using this water source", "Test all water sources", "Install certified filtration"],
            "professional_help": "URGENT: Contact certified water treatment professional immediately.",
            "additional_precautions": "Avoid all contact with contaminated water until professionally treated."
        }
    elif "10ppm" in concentration or "Medium" in concentration:
        return {
            "immediate_actions": ["Do not drink this water", "Use alternative water source", "Contact water authority"],
            "treatment_options": ["Boil water for 15 minutes", "Use activated carbon filter", "Consider reverse osmosis"],
            "health_risks": "Medium contamination level. Prolonged exposure may cause health issues.",
            "prevention_tips": ["Regular water testing", "Maintain filtration systems", "Monitor water quality"],
            "professional_help": "Contact water quality specialist for detailed analysis and treatment options.",
            "additional_precautions": "Use filtered or boiled water for all consumption until treated."
        }
    elif "0.1ppm" in concentration or "Low" in concentration:
        return {
            "immediate_actions": ["Boil water before drinking", "Use water filter", "Monitor for changes"],
            "treatment_options": ["Boiling", "Basic filtration", "Regular testing"],
            "health_risks": "Low level contamination. Monitor for any health changes.",
            "prevention_tips": ["Regular testing", "Maintain good water practices", "Monitor source quality"],
            "professional_help": "Consider professional testing for comprehensive analysis.",
            "additional_precautions": "Continue monitoring water quality regularly."
        }
    else:
        return {
            "immediate_actions": ["Water appears safe", "Continue regular testing", "Maintain good practices"],
            "treatment_options": ["Optional basic filtration", "Regular testing", "Source protection"],
            "health_risks": "No immediate health risks detected at this level.",
            "prevention_tips": ["Regular testing", "Source protection", "Maintain water systems"],
            "professional_help": "Routine testing recommended for ongoing monitoring.",
            "additional_precautions": "Continue regular testing to ensure water quality."
        }

def detect_test_tube_and_extract_color(image):
    """Detect test tube and average color using OpenCV"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        liquid_y = int(y + h * 0.1)
        liquid_h = int(h * 0.9)
        liquid_area = opencv_image[liquid_y:liquid_y + liquid_h, x:x + w]

        if liquid_area.size > 0:
            liquid_rgb = cv2.cvtColor(liquid_area, cv2.COLOR_BGR2RGB)
            pixels = liquid_rgb.reshape(-1, 3)
            filtered_pixels = [p for p in pixels if 150 < sum(p) < 650]
            if filtered_pixels:
                avg_r = int(np.mean([p[0] for p in filtered_pixels]))
                avg_g = int(np.mean([p[1] for p in filtered_pixels]))
                avg_b = int(np.mean([p[2] for p in filtered_pixels]))
                print(f"🎨 OpenCV Method - Extracted Color: RGB({avg_r}, {avg_g}, {avg_b})")
                return [avg_r, avg_g, avg_b]
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

        print(f"🔍 Received manual_color parameter: {manual_color}")
        detected_rgb = None

        if manual_color:
            try:
                detected_rgb = json.loads(manual_color)
                print(f"🎨 Manual Color - RGB: {detected_rgb}")
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Invalid manual color format, falling back to auto-detection.")
        
        if detected_rgb is None:
            detected_rgb = detect_test_tube_and_extract_color(image)

        if detected_rgb is None:
            print("⚠️  OpenCV detection failed, using fallback method...")
            width, height = image.size
            cropped = image.crop((width // 6, int(height * 0.1), int(width * 0.83), height))
            pixels = list(cropped.getdata())
            colored_pixels = [p for p in pixels if 150 < sum(p) < 650]
            if colored_pixels:
                avg_r = sum(p[0] for p in colored_pixels) // len(colored_pixels)
                avg_g = sum(p[1] for p in colored_pixels) // len(colored_pixels)
                avg_b = sum(p[2] for p in colored_pixels) // len(colored_pixels)
            else:
                avg_r = sum(p[0] for p in pixels) // len(pixels)
                avg_g = sum(p[1] for p in pixels) // len(pixels)
                avg_b = sum(p[2] for p in pixels) // len(pixels)
            detected_rgb = [avg_r, avg_g, avg_b]
            print(f"🎨 Fallback Method - RGB({avg_r}, {avg_g}, {avg_b})")

        # Normalize + convert to HSV for comparison
        normalized_rgb = normalize_rgb(detected_rgb)
        detected_hsv = rgb_to_hsv_tuple(normalized_rgb)

        # Compare against reference colors in HSV
        min_distance = float('inf')
        closest_match = None
        for ref in REFERENCE_COLORS:
            ref_hsv = rgb_to_hsv_tuple(ref['rgb'])
            distance = hsv_distance(detected_hsv, ref_hsv)
            print(f"Comparing detected HSV={detected_hsv} with ref {ref['metal']} HSV={ref_hsv} → distance={distance:.2f}")
            if distance < min_distance:
                min_distance = distance
                closest_match = ref

        # Safety check: ensure closest_match is not None (should always find a match from REFERENCE_COLORS)
        if closest_match is None:
            logger.warning(f"⚠️ closest_match is None, defaulting to Copper")
            closest_match = REFERENCE_COLORS[0]  # Default to first Copper entry
            print(f"🔄 Forced to use Copper: {closest_match}")
        
        # Force Copper - since REFERENCE_COLORS only contains Copper, this ensures consistency
        if closest_match.get('metal') != 'Copper':
            logger.warning(f"⚠️ Detected metal was {closest_match.get('metal')}, forcing to Copper")
            closest_match = REFERENCE_COLORS[0]
            print(f"🔄 Forced to Copper: {closest_match}")

        print(f"🎯 Selected match: {closest_match['metal']} - {closest_match['ppm']}")
        status, recommendation = get_status_and_recommendation(
            closest_match['metal'],
            closest_match['ppm']
        )

        # Generate AI recommendations if enabled, otherwise use basic recommendations
        if ai_service_enabled and ai_service:
            print("🤖 Generating AI recommendations...")
            ai_recommendations = ai_service.generate_smart_recommendations(
                closest_match['metal'],
                closest_match['ppm'],
                detected_rgb
            )
        else:
            print("📋 Using basic recommendations (AI disabled)...")
            ai_recommendations = get_basic_ai_recommendations(
                closest_match['metal'],
                closest_match['ppm']
            )

        result = AnalysisResponse(
            metal=closest_match['metal'],
            concentration=closest_match['ppm'],
            status=status,
            recommendation=recommendation,
            detected_rgb=detected_rgb,
            ai_recommendations=ai_recommendations
        )

        # Save to DB
        record = AnalysisResult(**result.model_dump())
        doc = record.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.analysis_results.insert_one(doc)

        print(f"✅ Final: {result.metal}, {result.concentration}, {result.status}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@api_router.get("/results", response_model=List[AnalysisResult])
async def get_results():
    try:
        results = await db.analysis_results.find({}, {"_id": 0}).sort("timestamp", -1).to_list(100)
        for r in results:
            if isinstance(r['timestamp'], str):
                r['timestamp'] = datetime.fromisoformat(r['timestamp'])
        return results
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")

@api_router.get("/")
async def root():
    return {
        "message": "AI-Enhanced Heavy Metal Detection Kit API",
        "ai_enabled": ai_service_enabled,
        "use_ai_flag": USE_AI,
        "features": [
            "HSV color-based detection (lighting independent)",
            "AI-powered recommendations" if ai_service_enabled else "Basic recommendations",
            "Smart safety advice",
            "Health risk assessment",
            "Treatment guidance"
        ],
        "note": "Set USE_AI=true in .env to enable AI features" if not USE_AI else "AI service enabled"
    }

# -----------------------------
# Startup & Shutdown
# -----------------------------
@app.on_event("startup")
async def startup_db_check():
    try:
        await client.list_database_names()
        print("✅ MongoDB is connected successfully!")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# -----------------------------
# Middleware & Router
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
# ASGI Application
# -----------------------------
application = app
server = app
