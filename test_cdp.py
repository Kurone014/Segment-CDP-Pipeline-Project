import requests
import json

# Test if server is running
try:
    # Send a test event
    response = requests.post('http://localhost:5000/track', 
        json={
            'event': 'page_view',
            'anonymousId': 'test123',
            'properties': {'page': '/home'}
        })
    print("✅ Server is running!")
    print(f"Response: {response.json()}")
except:
    print("❌ Server is not running. Start it with: python cdp_pipeline_fixed.py")