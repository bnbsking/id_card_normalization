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


def setup_model(cls_name: str, model_history_folder: str) -> BaseInference:
    # set threshold
    threshold = 0.5
    metrics_path = os.path.join(model_history_folder, "eval", "metrics.json")
    if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                threshold = metrics.get("best_threshold", threshold)

    # load model
    with open(os.environ["BASE_CFG_PATH"], "r") as f:
        cfg = yaml.safe_load(f)
    pretrain_weights = cfg["train"]["model"]["save_model_path"]
    model = getattr(infer, cls_name)(pretrain_weights, threshold)
    logger.info(f"Use model from {pretrain_weights} with threshold {threshold}")
    return model


setup_logging()
logger = logging.getLogger(__name__)

with open(os.environ["BASE_CFG_PATH"], "r") as f:
    model_cfgs = yaml.safe_load(f)["models"]
model_dict = {}
for model_cfg in model_cfgs:
    model_dict[model_cfg["key"]] = setup_model(model_cfg["cls_name"], model_cfg["model_history_folder"])


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
    Return example: {'pd_cls': 0, 'vflip_confidence': 0.12082499265670776}
    """
    meta_dict = json.loads(meta)
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        model = model_dict[meta_dict["model_key"]]
        result = model.inference(file_path)
    return result
    