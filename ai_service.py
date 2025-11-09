"""
AI Service Module for Enhanced Heavy Metal Detection
Integrates Google Gemini API for advanced image analysis and recommendations
"""

import os
import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Load .env file if not already loaded
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env', encoding='utf-8')

logger = logging.getLogger(__name__)

class AIService:
    """AI-powered analysis service using Google Gemini"""
    
    def __init__(self):
        """Initialize Gemini AI service"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. AI features will be disabled.")
            self.enabled = False
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.enabled = True
            logger.info("✅ Gemini AI service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini AI: {e}")
            self.enabled = False
    
    def generate_smart_recommendations(self, metal: str, concentration: str, 
                                     detected_rgb: List[int]) -> Dict:
        """
        Generate AI-powered recommendations and precautions based on detected color
        
        Args:
            metal: Detected metal type
            concentration: Detected concentration level
            detected_rgb: RGB values from traditional detection
            
        Returns:
            AI-generated recommendations and precautions
        """
        if not self.enabled:
            return self._get_basic_recommendations(metal, concentration)
        
        try:
            # Prepare prompt for AI recommendations
            prompt = self._create_recommendation_prompt(metal, concentration, detected_rgb)
            
            # Configure safety settings for scientific analysis
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Generate AI recommendations
            response = self.model.generate_content(prompt, safety_settings=safety_settings)
            
            # Parse AI response
            ai_recommendations = self._parse_recommendation_response(response.text)
            
            logger.info("✅ AI recommendations generated successfully")
            return ai_recommendations
            
        except Exception as e:
            logger.error(f"❌ AI recommendation generation failed: {e}")
            return self._get_basic_recommendations(metal, concentration)
    
    def _create_recommendation_prompt(self, metal: str, concentration: str, 
                                    detected_rgb: List[int]) -> str:
        """Create prompt for AI recommendations based on detected color"""
        
        return f"""
You are a water quality expert providing safety recommendations and precautions for heavy metal contamination.

DETECTED CONTAMINATION:
- Metal Type: {metal}
- Concentration Level: {concentration}
- Detected Color (RGB): {detected_rgb}

PROVIDE COMPREHENSIVE RECOMMENDATIONS INCLUDING:

1. IMMEDIATE SAFETY ACTIONS:
   - What to do right now
   - Emergency precautions
   - Who to contact

2. WATER TREATMENT OPTIONS:
   - Filtration methods
   - Boiling effectiveness
   - Professional treatment options
   - DIY solutions

3. HEALTH CONSIDERATIONS:
   - Health risks at this concentration
   - Symptoms to watch for
   - Vulnerable populations
   - Long-term effects

4. PREVENTION MEASURES:
   - How to prevent future contamination
   - Regular testing schedule
   - Water source protection
   - Home maintenance tips

5. WHEN TO SEEK PROFESSIONAL HELP:
   - Urgency level
   - What professionals to contact
   - What information to provide

RESPONSE FORMAT (JSON only):
{{
    "immediate_actions": ["action1", "action2", "action3"],
    "treatment_options": ["option1", "option2", "option3"],
    "health_risks": "detailed health information",
    "prevention_tips": ["tip1", "tip2", "tip3"],
    "professional_help": "when and how to get professional assistance",
    "additional_precautions": "extra safety measures"
}}

Make recommendations practical, scientifically accurate, and easy to understand for general public.
"""
    
    def _parse_recommendation_response(self, response_text: str) -> Dict:
        """Parse AI recommendation response and extract structured data"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse AI recommendation response: {e}")
            return self._create_default_recommendation_result()
    
    def _create_default_recommendation_result(self) -> Dict:
        """Create default recommendation result when parsing fails"""
        return {
            "immediate_actions": ["Do not consume the water", "Contact local water authority", "Use bottled water for drinking"],
            "treatment_options": ["Boil water for 10 minutes", "Use activated carbon filter", "Contact professional water treatment service"],
            "health_risks": "Heavy metal contamination can cause various health issues. Consult a healthcare provider if you have concerns.",
            "prevention_tips": ["Test water regularly", "Maintain plumbing systems", "Use certified water filters"],
            "professional_help": "Contact a certified water testing laboratory for detailed analysis and treatment recommendations.",
            "additional_precautions": "Avoid using contaminated water for cooking, drinking, or personal hygiene until properly treated."
        }
    
    def _get_basic_recommendations(self, metal: str, concentration: str) -> Dict:
        """Basic recommendations when AI is unavailable"""
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
    
    def _euclidean_distance(self, rgb1: List[int], rgb2: List[int]) -> float:
        """Calculate Euclidean distance between two RGB values"""
        return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

# Global AI service instance
ai_service = AIService()
