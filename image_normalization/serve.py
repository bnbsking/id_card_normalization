import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
from typing import Dict

import cv2
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import yaml

from src.exceptions import BaseCustomException
from src.logs import setup_logging
from src.main import Normalization


def get_args() -> Dict[str, str]:
    with open(os.environ["BASE_CFG_PATH"], "r") as f:
        args = yaml.safe_load(f)
    assert "detection_url" in args
    assert "classification_url" in args
    return args


setup_logging()
logger = logging.getLogger(__name__)
args = get_args()
PIPELINE = Normalization(**args)
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


class UploadRequest(BaseModel):
    meta: str

    @classmethod
    def as_form(cls, meta: str = Form(...)):
        return cls(meta=meta)


@app.post("/upload")
def upload(
        file: UploadFile = File(...),
        meta: UploadRequest = Depends(UploadRequest.as_form)
    ) -> Dict:
    with TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, file.filename)
        with open(img_path, "wb") as f:
            f.write(file.file.read())

        out_img = PIPELINE.run(img_path)
        out_img_path = os.path.join(tmpdir, f"out_{file.filename}")
        cv2.imwrite(out_img_path, out_img)
        
        image_bytes = Path(out_img_path).read_bytes()
        return Response(
            content=image_bytes,
            media_type="image/jpg",
        )
    

@app.post("/upload_full")
def upload_full(
        file: UploadFile = File(...),
        meta: UploadRequest = Depends(UploadRequest.as_form)   
    ) -> Dict[str, str]:
    with TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, file.filename)
        with open(img_path, "wb") as f:
            f.write(file.file.read())

        out_dict = PIPELINE.run_full(img_path)
        return out_dict
