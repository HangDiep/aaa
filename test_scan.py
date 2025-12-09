import requests
import json

try:
    print("🚀 Triggering /scan...")
    resp = requests.post("http://localhost:8000/notion/dynamic/scan", timeout=60)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        print("✅ Response:")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    else:
        print(f"❌ Error: {resp.text}")

except Exception as e:
    print(f"❌ Script Error: {e}")
