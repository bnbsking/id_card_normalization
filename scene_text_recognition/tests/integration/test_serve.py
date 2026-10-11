import json

import requests


def test_upload():
    img_path = "/dataset/images/pair_crop/0914/身份證/26452494799045_address_0.jpg"
    model_key = "svtrv2"

    url = "http://10.32.210.230:8063/upload"
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": model_key})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    print(result)  # 河南省封丘县城关乡刘村


if __name__ == "__main__":
    test_upload()