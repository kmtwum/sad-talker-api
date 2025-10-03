from fastapi import FastAPI, status, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate_onnx import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

import os

tts_service = os.getenv("TTS_SERVER")
facerender_batch_size = 16  # Increased from 10
sadtalker_paths = init_path("./checkpoints", os.path.join("/app", 'src/config'), "256", False, "crop")  # Increased from 10

preprocess_model = CropAndExtract(sadtalker_paths, "cuda")
audio_to_coeff = Audio2Coeff(sadtalker_paths, "cuda")
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, "cuda")

app = FastAPI()


class Words(BaseModel):
    words: str


@app.post("/pipeline")
async def predict_image(image: UploadFile = File(...), audio: UploadFile = File(...), use_enhancer: bool = True):
    # Save uploaded files
    pic_path = f"/app/img/{image.filename}"
    aud_path = f"/app/aud/{audio.filename}"
    out_path = "/app/output"

    with open(pic_path, "wb") as f:
        f.write(await image.read())
    with open(aud_path, "wb") as f:
        f.write(await audio.read())

    preprocess_mode = "crop"  # Changed from "full"

    first_frame_dir = os.path.join(out_path, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, preprocess_mode,
                                                                           source_image_flag=True)
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None
    batch = get_data(first_coeff_path, aud_path, "cuda", ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, out_path, 0, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, aud_path,
                               facerender_batch_size, None, None, None,
                               expression_scale=1, still_mode=True, preprocess=preprocess_mode)
    video_path = animate_from_coeff.generate_deploy(
        data, out_path, pic_path, crop_info,
        enhancer="gfpgan" if use_enhancer else None,
        background_enhancer=None,
        preprocess=preprocess_mode,
        skip_background_blend=True)

    return FileResponse(video_path, media_type="video/mp4", filename="result.mp4")


@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


@app.get("/health/inference")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
