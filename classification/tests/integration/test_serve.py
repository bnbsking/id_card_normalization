import json

import requests


def test_upload():
    img_path = "/dataset/images/crop/0914/台胞證/20569495559749.jpg"
    model_key = "vertical_flip_classification"
    
    url = "http://localhost:8062/upload"
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": model_key})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    print(result)
    """
    {'pd_cls': 0, 'vflip_confidence': 0.12082499265670776}
    """


if __name__ == "__main__":
    test_upload()