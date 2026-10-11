import json

import requests


def test_upload():
    img_path = "/dataset/images/remove_exif/0914/台胞證/20569495559749.jpg"
    model_key = "detection_model"

    url = "http://localhost:8061/upload"
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": model_key})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    print(result)
    """
    {'xmin': 63, 'ymin': 1383, 'xmax': 2793, 'ymax': 3131, 'confidence': 0.948}
    """


def test_pair_upload():
    img_path = "/dataset/images/crop_vflip/0914/駕駛證/20130494692933.jpg"
    model_key = "pair_detection_model"

    url = "http://localhost:8061/upload"
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": model_key})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    print(result)
    """
    {
        'boxes': [
            [910, 1421, 1521, 1527],
            [905, 777, 1394, 868],
            [895, 999, 1393, 1084],
            [1902, 785, 1986, 884],
            [1476, 999, 1979, 1089],
            [900, 420, 1280, 530]
        ],
        'confidence': [0.968, 0.94, 0.94, 0.939, 0.936, 0.931],
        'class_id': [3, 2, 2, 1, 2, 0]
    }
    """


if __name__ == "__main__":
    test_upload()
    test_pair_upload()
    