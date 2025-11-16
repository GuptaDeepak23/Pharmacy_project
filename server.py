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
    confidence: float = None  # Optional confidence score
    matchedType: str = None  # Optional match type (exact/approximate)

# -----------------------------
# Metal Detection Models (New System)
# -----------------------------
class MetalRange(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metal_name: str  # "Mercury" or "Lead"
    concentration_label: str  # "1–5 ppm", "More than 10 ppm", etc.
    r_min: int
    r_max: int
    g_min: int
    g_max: int
    b_min: int
    b_max: int

class DetectionRequest(BaseModel):
    r: int = Field(..., ge=0, le=255, description="Red value (0-255)")
    g: int = Field(..., ge=0, le=255, description="Green value (0-255)")
    b: int = Field(..., ge=0, le=255, description="Blue value (0-255)")

class DetectionResponse(BaseModel):
    metal: str
    concentration: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    matchedType: str  # "exact" or "approximate"
    input_rgb: List[int]

# -----------------------------
# Utility functions
# -----------------------------
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
# Metal Detection Functions (New System)
# -----------------------------
async def seed_metal_ranges():
    """Seed the database with metal range data"""
    metal_ranges_collection = db.metal_ranges
    
    # Check if data already exists
    existing_count = await metal_ranges_collection.count_documents({})
    if existing_count > 0:
        logger.info(f"Metal ranges already seeded ({existing_count} records)")
        return
    
    # Metal data as provided
    metal_data = [
        # Mercury (Hg)
        {"metal_name": "Mercury", "concentration_label": "More than 10 ppm", "r_min": 190, "r_max": 220, "g_min": 80, "g_max": 110, "b_min": 47, "b_max": 65},
        {"metal_name": "Mercury", "concentration_label": "5–10 ppm", "r_min": 220, "r_max": 240, "g_min": 110, "g_max": 130, "b_min": 47, "b_max": 65},
        {"metal_name": "Mercury", "concentration_label": "1–5 ppm", "r_min": 190, "r_max": 215, "g_min": 100, "g_max": 110, "b_min": 46, "b_max": 68},
        {"metal_name": "Mercury", "concentration_label": "0.01–1 ppm", "r_min": 165, "r_max": 189, "g_min": 90, "g_max": 110, "b_min": 30, "b_max": 65},
        {"metal_name": "Mercury", "concentration_label": "0.01 ppm", "r_min": 190, "r_max": 200, "g_min": 133, "g_max": 166, "b_min": 95, "b_max": 110},
        {"metal_name": "Mercury", "concentration_label": "Below 0.01 ppm", "r_min": 210, "r_max": 255, "g_min": 170, "g_max": 255, "b_min": 120, "b_max": 255},
        
        # Lead (Pb)
        {"metal_name": "Lead", "concentration_label": "More than 10 ppm", "r_min": 166, "r_max": 176, "g_min": 65, "g_max": 110, "b_min": 60, "b_max": 80},
        {"metal_name": "Lead", "concentration_label": "1–10 ppm", "r_min": 170, "r_max": 190, "g_min": 70, "g_max": 110, "b_min": 55, "b_max": 80},
        {"metal_name": "Lead", "concentration_label": "0.01–1 ppm", "r_min": 165, "r_max": 179, "g_min": 80, "g_max": 90, "b_min": 35, "b_max": 50},
        {"metal_name": "Lead", "concentration_label": "0.01 ppm", "r_min": 180, "r_max": 195, "g_min": 110, "g_max": 135, "b_min": 60, "b_max": 90},
        {"metal_name": "Lead", "concentration_label": "Below 0.01 ppm", "r_min": 210, "r_max": 255, "g_min": 170, "g_max": 255, "b_min": 120, "b_max": 255},
    ]
    
    # Insert metal ranges
    for metal_range in metal_data:
        metal_range["id"] = str(uuid.uuid4())
        await metal_ranges_collection.insert_one(metal_range)
    
    logger.info(f"✅ Seeded {len(metal_data)} metal ranges into database")

def check_exact_match(r: int, g: int, b: int, metal_range: dict) -> bool:
    """Check if RGB values exactly match a metal range"""
    return (
        metal_range["r_min"] <= r <= metal_range["r_max"] and
        metal_range["g_min"] <= g <= metal_range["g_max"] and
        metal_range["b_min"] <= b <= metal_range["b_max"]
    )

def calculate_distance_to_range(r: int, g: int, b: int, metal_range: dict) -> float:
    """Calculate distance from RGB to the midpoint of a metal range"""
    midpoint_r = (metal_range["r_min"] + metal_range["r_max"]) / 2
    midpoint_g = (metal_range["g_min"] + metal_range["g_max"]) / 2
    midpoint_b = (metal_range["b_min"] + metal_range["b_max"]) / 2
    
    # Manhattan distance as specified
    distance = abs(r - midpoint_r) + abs(g - midpoint_g) + abs(b - midpoint_b)
    return distance

async def detect_metal(r: int, g: int, b: int) -> DetectionResponse:
    """Detect metal and concentration from RGB values"""
    metal_ranges_collection = db.metal_ranges
    
    # Get all metal ranges from database
    all_ranges = await metal_ranges_collection.find({}).to_list(None)
    
    if not all_ranges:
        raise HTTPException(
            status_code=500,
            detail="Metal ranges not found in database. Please seed the database first."
        )
    
    # First, try exact match
    for metal_range in all_ranges:
        if check_exact_match(r, g, b, metal_range):
            # Calculate confidence based on how centered the value is in the range
            r_range = metal_range["r_max"] - metal_range["r_min"]
            g_range = metal_range["g_max"] - metal_range["g_min"]
            b_range = metal_range["b_max"] - metal_range["b_min"]
            
            r_center = (metal_range["r_min"] + metal_range["r_max"]) / 2
            g_center = (metal_range["g_min"] + metal_range["g_max"]) / 2
            b_center = (metal_range["b_min"] + metal_range["b_max"]) / 2
            
            # Calculate how close to center (0 = at edge, 1 = at center)
            r_confidence = 1.0 - abs(r - r_center) / (r_range / 2) if r_range > 0 else 1.0
            g_confidence = 1.0 - abs(g - g_center) / (g_range / 2) if g_range > 0 else 1.0
            b_confidence = 1.0 - abs(b - b_center) / (b_range / 2) if b_range > 0 else 1.0
            
            # Average confidence, clamped to [0.8, 1.0] for exact matches
            confidence = max(0.8, min(1.0, (r_confidence + g_confidence + b_confidence) / 3))
            
            return DetectionResponse(
                metal=metal_range["metal_name"],
                concentration=metal_range["concentration_label"],
                confidence=confidence,
                matchedType="exact",
                input_rgb=[r, g, b]
            )
    
    # No exact match found, find nearest match
    min_distance = float('inf')
    nearest_range = None
    
    for metal_range in all_ranges:
        distance = calculate_distance_to_range(r, g, b, metal_range)
        if distance < min_distance:
            min_distance = distance
            nearest_range = metal_range
    
    if nearest_range is None:
        raise HTTPException(
            status_code=500,
            detail="Could not determine nearest match"
        )
    
    # Calculate confidence for approximate match
    # Normalize distance (max possible distance is ~765 for RGB)
    max_possible_distance = 765.0  # 255 + 255 + 255
    normalized_distance = min_distance / max_possible_distance
    confidence = max(0.5, 1.0 - normalized_distance)  # Clamp to [0.5, 1.0]
    
    return DetectionResponse(
        metal=nearest_range["metal_name"],
        concentration=f"Closest Match: {nearest_range['concentration_label']}",
        confidence=round(confidence, 2),
        matchedType="approximate",
        input_rgb=[r, g, b]
    )

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

        # Use metal detection system
        detection_result = await detect_metal(detected_rgb[0], detected_rgb[1], detected_rgb[2])
        
        # Map detection result to analysis format
        metal_name = detection_result.metal
        concentration = detection_result.concentration
        
        # Remove "Closest Match: " prefix if present for cleaner display
        if concentration.startswith("Closest Match: "):
            concentration = concentration.replace("Closest Match: ", "")
        
        print(f"🎯 Detected: {metal_name} - {concentration} ({detection_result.matchedType})")
        status, recommendation = get_status_and_recommendation(
            metal_name,
            concentration
        )

        # Generate AI recommendations if enabled, otherwise use basic recommendations
        if ai_service_enabled and ai_service:
            print("🤖 Generating AI recommendations...")
            ai_recommendations = ai_service.generate_smart_recommendations(
                metal_name,
                concentration,
                detected_rgb
            )
        else:
            print("📋 Using basic recommendations (AI disabled)...")
            ai_recommendations = get_basic_ai_recommendations(
                metal_name,
                concentration
            )

        result = AnalysisResponse(
            metal=metal_name,
            concentration=concentration,
            status=status,
            recommendation=recommendation,
            detected_rgb=detected_rgb,
            ai_recommendations=ai_recommendations,
            confidence=detection_result.confidence,
            matchedType=detection_result.matchedType
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

@api_router.post("/detect-metal", response_model=DetectionResponse)
async def detect_metal_endpoint(request: DetectionRequest):
    """
    Detect metal and concentration from RGB values.
    
    - **r**: Red value (0-255)
    - **g**: Green value (0-255)
    - **b**: Blue value (0-255)
    
    Returns:
    - **metal**: Detected metal name (Mercury or Lead)
    - **concentration**: Concentration level
    - **confidence**: Confidence score (0.0-1.0)
    - **matchedType**: "exact" or "approximate"
    - **input_rgb**: Original RGB input
    """
    try:
        result = await detect_metal(request.r, request.g, request.b)
        logger.info(f"Metal detection: {result.metal} - {result.concentration} ({result.matchedType})")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting metal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting metal: {str(e)}")

@api_router.get("/metal-ranges", response_model=List[MetalRange])
async def get_metal_ranges():
    """Get all metal ranges from database"""
    try:
        metal_ranges_collection = db.metal_ranges
        ranges = await metal_ranges_collection.find({}).to_list(None)
        # Remove MongoDB _id field
        for r in ranges:
            r.pop("_id", None)
        return ranges
    except Exception as e:
        logger.error(f"Error fetching metal ranges: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metal ranges: {str(e)}")

@api_router.post("/seed-metal-ranges")
async def seed_metal_ranges_endpoint():
    """Manually seed metal ranges (admin endpoint)"""
    try:
        await seed_metal_ranges()
        return {"message": "Metal ranges seeded successfully"}
    except Exception as e:
        logger.error(f"Error seeding metal ranges: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error seeding metal ranges: {str(e)}")

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
            "Treatment guidance",
            "RGB-based metal detection (Mercury & Lead)"
        ],
        "note": "Set USE_AI=true in .env to enable AI features" if not USE_AI else "AI service enabled",
        "endpoints": {
            "detect_metal": "/api/detect-metal (POST)",
            "metal_ranges": "/api/metal-ranges (GET)",
            "seed_ranges": "/api/seed-metal-ranges (POST)"
        }
    }

# -----------------------------
# Startup & Shutdown
# -----------------------------
@app.on_event("startup")
async def startup_db_check():
    try:
        await client.list_database_names()
        print("✅ MongoDB is connected successfully!")
        # Seed metal ranges on startup
        await seed_metal_ranges()
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
