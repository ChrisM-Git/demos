"""
U.S. GOV Travel System - AI-Powered Travel Assistant
Multi-Model AI Gateway with Palo Alto Networks Security

Architecture:
- Orchestrator (GPT-4o): Plans and calls tools
- Researcher (Claude/GPT-4o): Creates detailed itineraries
- Calculator (GPT-4o-mini): Budget calculations

All requests route through LiteLLM proxy with PANW Prisma AIRS guardrails
"""

import os
import json
import hashlib
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

PORT = 5200

# LiteLLM Proxy Configuration
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")

# Model assignments (defined in litellm_config.yaml)
MODELS = {
    "orchestrator": "orchestrator",
    "researcher": "researcher",
    "calculator": "calculator",
}

# =============================================================================
# DYNAMIC DATA GENERATION - Realistic prices for any city
# =============================================================================

def generate_flights(origin: str, destination: str, date: str) -> dict:
    """Generate realistic flight data based on route."""
    route_hash = int(hashlib.md5(f"{origin}{destination}".encode()).hexdigest()[:8], 16)
    random.seed(route_hash)

    dest_lower = destination.lower()

    # Determine base price by distance category
    if any(city in dest_lower for city in ['tokyo', 'paris', 'london', 'rome', 'sydney', 'beijing', 'dubai']):
        base_price = random.randint(650, 850)
        duration_base = random.randint(10, 14)
    elif any(city in dest_lower for city in ['chicago', 'new york', 'boston', 'miami', 'atlanta']):
        base_price = random.randint(280, 420)
        duration_base = random.randint(4, 6)
    else:
        base_price = random.randint(150, 280)
        duration_base = random.randint(1, 3)

    airlines = [("United", "UA"), ("Delta", "DL"), ("American", "AA"), ("Alaska", "AS")]
    flights = []

    for i, (airline, code) in enumerate(airlines):
        stops = 0 if i < 2 else 1
        price = base_price + random.randint(-80, 120) + (0 if stops == 0 else -50)
        duration = duration_base + (1 if stops == 1 else 0)
        departure_hour = 6 + (i * 3)

        flights.append({
            "airline": airline,
            "flight": f"{code}{random.randint(100, 999)}",
            "departure": f"{departure_hour:02d}:{random.randint(0, 59):02d}",
            "price": max(price, 120),
            "stops": stops,
            "duration": f"{duration}h {random.randint(0, 59)}m"
        })

    flights.sort(key=lambda x: x['price'])
    return {"flights": flights, "origin": origin, "destination": destination, "date": date}


def generate_hotels(city: str, checkin: str, checkout: str, budget: str = "moderate") -> dict:
    """Generate realistic hotel data."""
    city_hash = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    random.seed(city_hash)

    price_ranges = {
        "luxury": [380, 420, 495, 550, 625],
        "budget": [65, 85, 95, 110, 135],
        "moderate": [145, 175, 195, 225, 265]
    }

    templates = {
        "luxury": [
            ("The Ritz-Carlton {city}", 4.8),
            ("Four Seasons {city}", 4.9),
            ("{city} Luxury Suites", 4.7),
            ("Grand Hyatt {city}", 4.6),
            ("The St. Regis {city}", 4.8)
        ],
        "budget": [
            ("Best Western {city}", 4.0),
            ("Comfort Inn {city}", 3.9),
            ("{city} Budget Hotel", 3.8),
            ("Days Inn {city}", 3.7),
            ("La Quinta Inn {city}", 4.1)
        ],
        "moderate": [
            ("{city} Marriott Downtown", 4.3),
            ("Hilton {city} Center", 4.4),
            ("The Westin {city}", 4.5),
            ("Hyatt Regency {city}", 4.4),
            ("DoubleTree by Hilton {city}", 4.2)
        ]
    }

    base_prices = price_ranges.get(budget, price_ranges["moderate"])
    hotel_list = templates.get(budget, templates["moderate"])

    hotels = []
    for i, (template, rating) in enumerate(hotel_list):
        hotels.append({
            "name": template.format(city=city.title()),
            "rating": round(rating + random.uniform(-0.2, 0.2), 1),
            "price_per_night": base_prices[i] + random.randint(-25, 35),
            "address": f"{random.randint(100, 9999)} {random.choice(['Main', 'Market', 'Broadway', 'Park'])} St, {city.title()}"
        })

    return {"hotels": hotels, "city": city, "checkin": checkin, "checkout": checkout}


def generate_activities(city: str, interests: list) -> dict:
    """Generate realistic activities."""
    city_hash = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    random.seed(city_hash)

    activity_types = {
        "culture": [
            ("{city} Art Museum", "2-3 hours", 18, 4.6),
            ("{city} History Museum", "3 hours", 15, 4.5),
            ("Historic {city} Walking Tour", "2 hours", 25, 4.7),
        ],
        "food": [
            ("{city} Food Tour", "3 hours", 75, 4.8),
            ("Local Cuisine Tasting", "2 hours", 50, 4.7),
            ("{city} Brewery Tour", "2.5 hours", 45, 4.6),
        ],
        "modern": [
            ("{city} Skyline Observatory", "1.5 hours", 28, 4.6),
            ("{city} Modern Art Gallery", "2 hours", 20, 4.5),
            ("Downtown {city} Shopping", "3 hours", 0, 4.3),
        ],
        "nature": [
            ("{city} City Park", "2 hours", 0, 4.6),
            ("{city} Botanical Gardens", "1.5 hours", 12, 4.7),
            ("Waterfront Walk in {city}", "2 hours", 0, 4.5),
        ]
    }

    activities = []
    for interest in interests:
        if interest in activity_types:
            for template, duration, base_price, rating in activity_types[interest]:
                activities.append({
                    "name": template.format(city=city.title()),
                    "duration": duration,
                    "price": max(base_price + random.randint(-5, 15), 0),
                    "rating": round(rating + random.uniform(-0.2, 0.2), 1),
                })

    return {"activities": activities, "city": city, "interests": interests}


def get_weather(city: str, date: str) -> dict:
    """Get weather info."""
    return {
        "city": city,
        "date": date,
        "temperature": "65°F / 18°C",
        "condition": "Partly cloudy",
        "humidity": "60%"
    }


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin city"},
                    "destination": {"type": "string", "description": "Destination city"},
                    "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)"}
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "checkin": {"type": "string"},
                    "checkout": {"type": "string"},
                    "budget": {"type": "string", "enum": ["budget", "moderate", "luxury"]}
                },
                "required": ["city", "checkin", "checkout"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_activities",
            "description": "Search for activities",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "interests": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["city", "interests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["city", "date"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute tool and return results."""
    tools = {
        "search_flights": lambda a: generate_flights(a["origin"], a["destination"], a["date"]),
        "search_hotels": lambda a: generate_hotels(a["city"], a["checkin"], a["checkout"], a.get("budget", "moderate")),
        "search_activities": lambda a: generate_activities(a["city"], a["interests"]),
        "get_weather": lambda a: get_weather(a["city"], a["date"])
    }
    return tools.get(tool_name, lambda a: {})(arguments)


# =============================================================================
# AI MODEL CALLS
# =============================================================================

def call_model(model_type: str, messages: list, tools: list = None, enable_guardrails: bool = True) -> dict:
    """Call AI model through LiteLLM proxy with optional PANW security."""
    client = OpenAI(
        api_key=LITELLM_MASTER_KEY,
        base_url=LITELLM_PROXY_URL + "/v1"
    )

    temp_map = {"orchestrator": 0.3, "researcher": 0.7, "calculator": 0.5}

    kwargs = {
        "model": MODELS[model_type],
        "messages": messages,
        "temperature": temp_map.get(model_type, 0.7),
        "max_tokens": 4000 if model_type == "researcher" else 2000,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # Enable PANW Prisma AIRS guardrails only for orchestrator (user input validation)
    # Skip for researcher/calculator to improve response time
    if enable_guardrails and model_type == "orchestrator":
        kwargs["extra_body"] = {
            "guardrails": ["panw-prisma-airs-pre", "panw-prisma-airs-post"]
        }

    return client.chat.completions.create(**kwargs)


# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================

def process_travel_request(user_message: str, conversation_history: list = None) -> dict:
    """Process travel request using multi-model AI."""
    if conversation_history is None:
        conversation_history = []

    models_used = []
    tools_called = []
    collected_data = {}

    # Step 1: Orchestrator gathers data
    system_prompt = """You are a travel planning assistant. When users provide specific travel requests with origin, destination, and dates, use the available tools to gather information. Call all relevant tools to get flights, hotels, activities, and weather data."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    models_used.append({"name": "Orchestrator", "model": MODELS["orchestrator"]})

    # Tool calling loop
    for iteration in range(3):
        try:
            response = call_model("orchestrator", messages, tools=TOOLS)
            assistant_message = response.choices[0].message
        except Exception as e:
            print(f"\n❌ ORCHESTRATOR ERROR: {e}")
            error_str = str(e)
            if 'panw_prisma_airs_blocked' in error_str or 'guardrail_violation' in error_str:
                return {
                    "response": None,
                    "models_used": models_used,
                    "tools_called": tools_called,
                    "error": error_str
                }
            import traceback
            traceback.print_exc()
            break

        if assistant_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in assistant_message.tool_calls
                ]
            })

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_name, arguments)
                collected_data[tool_name] = result
                tools_called.append({"name": tool_name})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            break

    # Step 2: Calculator for budget
    budget_breakdown = ""
    if collected_data:
        models_used.append({"name": "Calculator", "model": MODELS["calculator"]})

        calc_messages = [
            {"role": "system", "content": "You are a precise calculator. Calculate trip costs clearly."},
            {"role": "user", "content": f"Calculate total trip cost from this data:\n{json.dumps(collected_data, indent=2)}\n\nProvide: flight cost (best value), hotel cost (total nights), activities cost, estimated daily expenses ($80/day)."}
        ]

        try:
            calc_response = call_model("calculator", calc_messages)
            budget_breakdown = calc_response.choices[0].message.content
        except Exception as e:
            print(f"\n❌ CALCULATOR ERROR: {e}")
            import traceback
            traceback.print_exc()
            budget_breakdown = "Budget calculation unavailable."

    # Step 3: Researcher creates itinerary
    models_used.append({"name": "Researcher", "model": MODELS["researcher"]})

    # Extract specific options
    flights = collected_data.get('search_flights', {}).get('flights', [])
    hotels = collected_data.get('search_hotels', {}).get('hotels', [])
    activities = collected_data.get('search_activities', {}).get('activities', [])

    flight = flights[0] if flights else None
    hotel = hotels[1] if len(hotels) > 1 else (hotels[0] if hotels else None)

    # Build simple, explicit prompt
    prompt = f"""Create a travel itinerary for: {user_message}

RECOMMENDED FLIGHT:
Airline: {flight['airline'] if flight else 'N/A'}
Flight: {flight['flight'] if flight else 'N/A'}
Cost: ${flight['price'] if flight else 0}
Departure: {flight['departure'] if flight else 'N/A'}
Duration: {flight['duration'] if flight else 'N/A'}

RECOMMENDED HOTEL:
Name: {hotel['name'] if hotel else 'N/A'}
Cost: ${hotel['price_per_night'] if hotel else 0} per night
Rating: {hotel['rating'] if hotel else 0} stars

AVAILABLE ACTIVITIES:
"""
    for act in activities[:6]:
        prompt += f"- {act['name']}: ${act['price']} ({act['duration']})\n"

    prompt += f"\n{budget_breakdown}\n\n"
    prompt += "Create a detailed day-by-day itinerary with introduction, flight/hotel recommendations, daily activities, practical tips, and budget summary. Use the exact prices shown above."

    research_messages = [
        {"role": "system", "content": "You are a helpful travel assistant. Create detailed itineraries using the information provided. When prices are given, include them exactly as stated - these are example prices for planning purposes, not actual quotes or financial advice."},
        {"role": "user", "content": prompt}
    ]

    try:
        research_response = call_model("researcher", research_messages)
        final_response = research_response.choices[0].message.content
    except Exception as e:
        print(f"\n❌ RESEARCHER ERROR: {e}")
        import traceback
        traceback.print_exc()
        final_response = "I encountered an error creating your itinerary. Please try again."

    return {
        "response": final_response,
        "models_used": models_used,
        "tools_called": tools_called
    }


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'travel.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        result = process_travel_request(user_message, conversation_history)

        if result.get("error"):
            return jsonify({
                "success": False,
                "error": result["error"],
                "models_used": result.get("models_used", []),
                "tools_called": result.get("tools_called", [])
            })

        return jsonify({
            "success": True,
            "response": result["response"],
            "models_used": result["models_used"],
            "tools_called": result["tools_called"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "port": PORT})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🌍 U.S. GOV Travel System - Multi-Model AI Demo")
    print("=" * 60)
    print(f"\n📍 Server: http://localhost:{PORT}")
    print(f"🛡️  Security: {LITELLM_PROXY_URL}")
    print(f"   • Palo Alto Networks Prisma AIRS enabled")
    print(f"\n🤖 Models:")
    for role, model in MODELS.items():
        print(f"   • {role}: {model}")
    print("\n" + "=" * 60)

    app.run(host='0.0.0.0', port=PORT, debug=False)
