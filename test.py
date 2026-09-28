import requests
import json

# Make sure your FastAPI server is running on port 8000
url = "http://127.0.0.1:8000/evaluate"

payload = {
    "prompt": "What is the difference between nuclear fission and nuclear fusion?"
}

print(f"Sending prompt: '{payload['prompt']}'")
print("Evaluating (this will take a few seconds as it searches and queries the LLM)...")

try:
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    
    # Print the formatted JSON response
    result = response.json()
    print("\n=== EVALUATION RESULTS ===")
    print(json.dumps(result, indent=2))

except requests.exceptions.RequestException as e:
    print(f"\nAPI Request Failed: {e}")