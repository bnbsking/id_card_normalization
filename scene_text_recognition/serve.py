from functools import partial
import json
import logging
import os
from tempfile import TemporaryDirectory
import traceback
from typing import Callable

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
import yaml

from src.exceptions import BaseCustomException
from src.infer.infer import single_inference_threshold, joint_inference_threshold
from src.logs import setup_logging


def setup_model_single(model_path: str) -> Callable:
    model = partial(single_inference_threshold, model_path=model_path)
    return model


def setup_model_joint(model_path1: str, model_path2: str) -> Callable:
    model = partial(joint_inference_threshold, model_path1=model_path1, model_path2=model_path2)
    return model


setup_logging()
logger = logging.getLogger(__name__)


with open(os.environ.get("BASE_CFG_PATH", "/app/cfgs/cfg.yaml"), "r") as f:
    model_cfgs = yaml.safe_load(f)["models"]
model_dict = {}
for model_cfg in model_cfgs:
    #model_dict[model_cfg["key"]] = setup_model_single(model_cfg["model_path"])
    model_dict[model_cfg["key"]] = setup_model_joint(model_cfg["model_path1"], model_cfg["model_path2"])


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
    ) -> str:
    meta_dict = json.loads(meta)
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        model = model_dict[meta_dict["model_key"]]
        result = model(infer_img=file_path)
    return result
