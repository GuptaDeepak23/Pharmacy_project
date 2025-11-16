# Metal Detection API Documentation

## Overview

The Metal Detection System identifies metals (Mercury or Lead) and their concentration levels based on RGB color values.

## API Endpoints

### 1. POST `/api/detect-metal`

Detect metal and concentration from RGB values.

**Request Body:**
```json
{
  "r": 180,
  "g": 95,
  "b": 60
}
```

**Response:**
```json
{
  "metal": "Lead",
  "concentration": "0.01–1 ppm",
  "confidence": 0.92,
  "matchedType": "exact",
  "input_rgb": [180, 95, 60]
}
```

**Response Fields:**
- `metal`: Detected metal name ("Mercury" or "Lead")
- `concentration`: Concentration level (e.g., "1–5 ppm", "Closest Match: 0.01–1 ppm")
- `confidence`: Confidence score (0.0-1.0)
- `matchedType`: "exact" or "approximate"
- `input_rgb`: Original RGB input values

### 2. GET `/api/metal-ranges`

Get all metal ranges stored in the database.

**Response:**
```json
[
  {
    "id": "uuid",
    "metal_name": "Mercury",
    "concentration_label": "1–5 ppm",
    "r_min": 190,
    "r_max": 215,
    "g_min": 100,
    "g_max": 110,
    "b_min": 46,
    "b_max": 68
  },
  ...
]
```

### 3. POST `/api/seed-metal-ranges`

Manually seed the database with metal range data (useful if data was deleted).

**Response:**
```json
{
  "message": "Metal ranges seeded successfully"
}
```

## Detection Logic

### Exact Match
If the input RGB values fall within any metal range:
- `R_MIN <= r <= R_MAX`
- `G_MIN <= g <= G_MAX`
- `B_MIN <= b <= B_MAX`

Then return that metal and concentration with `matchedType: "exact"`.

### Nearest Match
If no exact match is found:
1. Calculate distance to each metal range midpoint:
   ```
   distance = abs(r - midpoint_R) + abs(g - midpoint_G) + abs(b - midpoint_B)
   ```
2. Choose the metal range with minimum distance
3. Return with `matchedType: "approximate"` and confidence based on distance

## Supported Metals & Concentrations

### Mercury (Hg)
- More than 10 ppm
- 5–10 ppm
- 1–5 ppm
- 0.01–1 ppm
- 0.01 ppm
- Below 0.01 ppm

### Lead (Pb)
- More than 10 ppm
- 1–10 ppm
- 0.01–1 ppm
- 0.01 ppm
- Below 0.01 ppm

## Database Schema

**Collection:** `metal_ranges`

```json
{
  "id": "string (UUID)",
  "metal_name": "string (Mercury or Lead)",
  "concentration_label": "string",
  "r_min": "integer (0-255)",
  "r_max": "integer (0-255)",
  "g_min": "integer (0-255)",
  "g_max": "integer (0-255)",
  "b_min": "integer (0-255)",
  "b_max": "integer (0-255)"
}
```

## Example Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/detect-metal",
    json={"r": 180, "g": 95, "b": 60}
)
print(response.json())
```

### JavaScript/Fetch
```javascript
fetch('http://localhost:8000/api/detect-metal', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ r: 180, g: 95, b: 60 })
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL
```bash
curl -X POST http://localhost:8000/api/detect-metal \
  -H "Content-Type: application/json" \
  -d '{"r": 180, "g": 95, "b": 60}'
```

## Auto-Seeding

The database is automatically seeded with metal range data on server startup. If you need to re-seed manually, use the `/api/seed-metal-ranges` endpoint.

