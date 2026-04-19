import requests
import json

test_data = {
    "data": [
        {
            "Region_Code": "IN_KL_TVM",
            "Dwelling_Type": "Independent House",
            "Num_Occupants": 2,
            "House_Area (sqft)": 2458,
            "Appliance_Score": 6,
            "Connected_Load(kw)": 7.21,
            "Temperature_C": 29.52,
            "Humidity (%)": 83.53,
            "Expected_Energy(kwh)": "12.82 kWh",
            "Actual_Energy(kwh)": "25.0 kWh"
        }
    ]
}

url = "http://localhost:5000/predict"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(test_data), headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
