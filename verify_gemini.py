import os
from google import genai
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("DEFAULT_MODEL", "gemini-2.5-flash")

print(f"Loaded API key from environment: {api_key[:8] if api_key else 'None'}...")
print(f"Using model: {model_name}")

# Initialize Client with the loaded API key
client = genai.Client(api_key=api_key)

try:
    # Send a simple prompt to the model
    response = client.models.generate_content(
        model=model_name,
        contents="Hello"
    )
    print("API Response:")
    print(response.text)
except Exception as e:
    print("API Call Failed!")
    print(f"Exception Type: {type(e)}")
    print(f"Exception Message: {e}")
    if hasattr(e, "details"):
        print(f"Detailed Error: {getattr(e, 'details')}")
