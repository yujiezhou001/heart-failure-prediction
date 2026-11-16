#!/usr/bin/env python
"""Send one patient to the running service and print the answer.

Start the service first:  uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Then run:                 uv run python predict-test.py

To test the cloud deployment, pass its address:
                          uv run python predict-test.py https://<app-name>.fly.dev
"""

import sys

import requests


host = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696'
url = f'{host}/predict'

patient_id = 'patient-563'
patient = {
    'age': 55,
    'sex': 'm',
    'chest_pain_type': 'asy',
    'resting_bp': 135,
    'cholesterol': 204,
    'fasting_bs': 1,
    'resting_ecg': 'st',
    'max_hr': 126,
    'exercise_angina': 'y',
    'oldpeak': 1.1,
    'st_slope': 'flat',
}


response = requests.post(url, json=patient).json()
print(response)

if response['heart_disease']:
    print(f'refer {patient_id} for follow-up tests')
else:
    print(f'no follow-up needed for {patient_id}')
