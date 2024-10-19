import requests


def single_request():
    url = "http://127.0.0.1:8000/generate/"
    payload = {
        "prompt": "white shark",
        "num_inference_steps": 50
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(data["message"])
        else:
            print(f"Failed: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    single_request()