import cv2

from src.main import Normalization


class TestNormalization:
    def test_detection_crop(self):
        url = "http://10.32.210.230:8061/upload"
        img_path = "/app/tests/integration/input/main/remove_exif.jpg"
        output_path = "/app/tests/integration/output/main/remove_exif_detection_crop.jpg"

        obj = Normalization(detection_url=url, classification_url="")
        out = obj.detection_crop(url=url, img_path=img_path)
        cv2.imwrite(output_path, out)

    def test_vflip(self):
        url = "http://10.32.210.230:8062/upload"
        img_path = "/app/tests/integration/input/main/detection_crop.jpg"
        output_path = "/app/tests/integration/output/main/detection_crop_vflip.jpg"

        obj = Normalization(detection_url="", classification_url=url)
        out = obj.vflip(url=url, img_path=img_path)
        cv2.imwrite(output_path, out)

    def test_run(self):
        img_path = "/app/tests/integration/input/main/raw.jpg"
        output_path = "/app/tests/integration/output/main/raw_main.jpg"
        detection_url = "http://10.32.210.230:8061/upload"
        classification_url = "http://10.32.210.230:8062/upload"
        
        obj = Normalization(detection_url=detection_url, classification_url=classification_url)
        out = obj.run(img_path=img_path)
        cv2.imwrite(output_path, out)


if __name__ == "__main__":
    obj = TestNormalization()
    #obj.test_detection_crop()
    #obj.test_vflip()
    obj.test_run()
    