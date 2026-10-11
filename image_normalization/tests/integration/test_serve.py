import requests


def test_upload():
    input_path = "/app/tests/integration/input/main/raw.jpg"
    output_path = "/app/tests/integration/output/main/raw_serve.jpg"
    url = "http://localhost:8060/upload"

    files = {"file": open(input_path, "rb")}
    data = {"meta": "meta_value"}
    response = requests.post(url, files=files, data=data)
    with open(output_path, "wb") as f:
        f.write(response.content)


def test_upload_full():
    input_path = "/app/tests/integration/input/main/20118032081733.jpeg"
    url = "http://localhost:8060/upload_full"

    files = {"file": open(input_path, "rb")}
    data = {"meta": "meta_value"}
    response = requests.post(url, files=files, data=data)
    print(response.json())  # {'id_num': '08408452', 'date': '1983.05.03', 'name': '江秋南', 'gender': '男'}


if __name__ == "__main__":
    #test_upload()
    test_upload_full()
