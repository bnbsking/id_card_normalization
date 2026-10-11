from src.crop_card_to_json import main


def test_main():
    detection_url = "http://10.32.210.230:8061/upload"
    img_path = "/app/tests/integration/input/main/detection_crop.jpg"
    categories = ["name", "gender", "date", "id_num", "address"]
    str_url = "http://10.32.210.230:8063/upload"
    result = main(detection_url, img_path, categories, str_url)
    print(result)  # {'date': '1980.10.28', 'id_num': '07915126', 'name': '连经侠', 'gender': '男'}


if __name__ == "__main__":
    test_main()
