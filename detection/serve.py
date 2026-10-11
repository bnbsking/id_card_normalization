import json
import logging
import os
from tempfile import TemporaryDirectory
import traceback
from typing import Dict

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
import yaml

from src.exceptions import BaseCustomException
from src.infer import infer
from src.infer.base import BaseInference
from src.logs import setup_logging


def setup_model(cls_name: str, model_folder: str) -> BaseInference:
    # set threshold
    threshold = 0.5
    metrics_path = os.path.join(model_folder, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            threshold = metrics.get("best_threshold", threshold)

    # load model
    pretrain_weights = os.path.join(model_folder, "checkpoint_best_ema.pth")
    model = getattr(infer, cls_name)(pretrain_weights, threshold)
    logger.info(f"Use model from {pretrain_weights} with threshold {threshold}")
    return model


setup_logging()
logger = logging.getLogger(__name__)


with open(os.environ.get("BASE_CFG_PATH", "/app/cfgs/cfg.yaml"), "r") as f:
    model_cfgs = yaml.safe_load(f)["models"]
model_dict = {}
for model_cfg in model_cfgs:
    model_dict[model_cfg["key"]] = setup_model(model_cfg["cls_name"], model_cfg["model_folder"])


logger.info("Start serving API...")
app = FastAPI()


@app.exception_handler(BaseCustomException)
async def custom_exception_handler(request: Request, exc: BaseCustomException) -> str:
    logger.error(f"[Client error] {exc.message}")
    return JSONResponse(
        status_code=exc.code,
        content=f"[Client error] {exc.message}"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> str:
    tb = traceback.format_exc()
    logger.error(f"[Internal server error] {tb}, {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=f"[Internal server error] {str(exc)}"
    )


@app.post("/upload")
def upload(
        file: UploadFile = File(...),
        meta: str = Form(...)
    ) -> Dict:
    """
    Return example:
        - if meta_dict["model_key"] == "detection_model": 
            {'xmin': 97, 'ymin': 1378, 'xmax': 2812, 'ymax': 3131, 'confidence': 0.777}
        - if meta_dict["model_key"] == "pair_detection_model":
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
    meta_dict = json.loads(meta)
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        model = model_dict[meta_dict["model_key"]]
        result = model.inference(file_path)
    return result
