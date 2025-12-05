import requests
import json

BASE_URL = "http://localhost:8000"

def print_response(title, response):

    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_api():

    print("\n🧪 TESTING PT100 SENSOR ANOMALY DETECTION API")
    print("="*70)


    response = requests.get(f"{BASE_URL}/")
    print_response("TEST 1: Root Endpoint", response)


    response = requests.get(f"{BASE_URL}/health")
    print_response("TEST 2: Health Check", response)


    response = requests.get(f"{BASE_URL}/records")
    print_response("TEST 3: Get All Records", response)


    response = requests.get(f"{BASE_URL}/records/1")
    print_response("TEST 4: Get Record by ID (ID=1)", response)


    new_record = {
        "value": 26.5,
        "timestamp": "2024-12-05T02:30:00"
    }
    response = requests.post(f"{BASE_URL}/records", json=new_record)
    print_response("TEST 5: Create New Record", response)
    created_id = response.json().get("id") if response.status_code == 201 else None


    if created_id:
        update_data = {"value": 27.0}
        response = requests.put(f"{BASE_URL}/records/{created_id}", json=update_data)
        print_response(f"TEST 6: Update Record (ID={created_id})", response)


    normal_prediction = {"value": 24.5}
    response = requests.post(f"{BASE_URL}/predict", json=normal_prediction)
    print_response("TEST 7: Predict NORMAL Value (24.5°C)", response)


    anomaly_prediction = {"value": 150.0}
    response = requests.post(f"{BASE_URL}/predict", json=anomaly_prediction)
    print_response("TEST 8: Predict ANOMALY Value (150.0°C)", response)


    batch_predictions = [
        {"value": 24.5},
        {"value": 150.0},
        {"value": 25.1},
        {"value": 10.0}
    ]
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_predictions)
    print_response("TEST 9: Batch Prediction", response)


    response = requests.get(f"{BASE_URL}/statistics")
    print_response("TEST 10: Get Statistics", response)


    if created_id:
        response = requests.delete(f"{BASE_URL}/records/{created_id}")
        print_response(f"TEST 11: Delete Record (ID={created_id})", response)


    response = requests.get(f"{BASE_URL}/records/9999")
    print_response("TEST 12: 404 Error - Record Not Found", response)

    print("\n" + "="*70)
    print("✅ API TESTING COMPLETE!")
    print("="*70)
    print("\n💡 View interactive documentation at:")
    print(f"   📍 {BASE_URL}/docs")
    print(f"   📍 {BASE_URL}/redoc")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("   Make sure the server is running:")
        print("   python api/main.py")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print()
