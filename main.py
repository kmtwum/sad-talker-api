from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger
import requests

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate_onnx import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

import os
import torch
import gc
import uuid

facerender_batch_size = 32  # Reduced to prevent OOM
sadtalker_paths = init_path("./checkpoints", os.path.join("/app", 'src/config'), "256", False, "full")

preprocess_model = CropAndExtract(sadtalker_paths, "cuda")
audio_to_coeff = Audio2Coeff(sadtalker_paths, "cuda")
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, "cuda")

app = FastAPI()
from dotenv import load_dotenv

load_dotenv()


def cleanup_files(temp_files: list):
    """Background task to clean up files after a response is sent"""
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    torch.cuda.empty_cache()
    gc.collect()


class Words(BaseModel):
    words: str


def generate_tts(text: str, tts_preference: str, out_path: str, session_id: str, user_id: str = None,
                 voice_source: str = None):
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url, json={
            "text": text,
            "split_sentences": False,
            "source_aud": voice_source,
            "clone": user_id,
            "model": "tts_models/multilingual/multi-dataset/xtts_v2"
        })
        tts_response.raise_for_status()
        audio_path = f"{out_path}/{session_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = get_secret_key("ELEVENLABS_API_KEY_FILE")
        voice_id = os.getenv("VOICE_ID")

        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
        )

        print("Saving audio file...")
        audio_path = f"{out_path}/{session_id}.mp3"
        with open(audio_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"Audio file saved to {audio_path}")

        return audio_path


@app.post("/generate")
async def predict_image(
        background_tasks: BackgroundTasks,
        text: str = Form(...),
        user_id: str = Form(...),
        tts_preference: str = Form("coqui"),
        source_img: str = Form(None),
        source_aud: str = Form(None),
        use_enhancer: bool = False):
    out_path = f"/app/output/{user_id}"
    os.makedirs(out_path, exist_ok=True)

    preprocess_mode = "full"
    temp_files = []

    # Create session
    session_id = uuid.uuid4()

    # Save image
    if user_id:
        img_path = f"/app/user_img/{user_id}.jpg"
        if not os.path.exists(img_path):
            # download image from url
            print("Downloading image...")
            try:
                gcp_base = get_secret_key("GCP_BASE_FILE")
                response = requests.get(f"{gcp_base}/{source_img}")
                response.raise_for_status()
                print("Image downloaded successfully!")
                with open(img_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading image: {e}")
                img_path = "/app/img/avatar.jpg"
    else:
        img_path = "/app/img/avatar.png"

    # Generate TTS
    audio_path = generate_tts(text, tts_preference, out_path, str(session_id), user_id=user_id, voice_source=source_aud)
    temp_files.append(audio_path)

    meta_dir = os.path.join(out_path, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(img_path, meta_dir, preprocess_mode)
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None
    batch = get_data(first_coeff_path, audio_path, "cuda", ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, out_path, 0, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               facerender_batch_size, None, None, None,
                               expression_scale=1, still_mode=True, preprocess=preprocess_mode)
    video_path = animate_from_coeff.generate_deploy(
        data, out_path, img_path, crop_info,
        enhancer="gfpgan" if use_enhancer else None,
        background_enhancer=None,
        preprocess=preprocess_mode,
        skip_background_blend=True)

    populate_temp_files(temp_files, out_path, user_id, str(session_id))

    # Schedule cleanup after the response is sent
    background_tasks.add_task(cleanup_files, temp_files)

    return FileResponse(video_path, media_type="video/mp4", filename="result.mp4")


def get_secret_key(secret):
    key_file = os.getenv(secret)
    if key_file:
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
            return api_key
    return None


def populate_temp_files(temp_files: list, out_path, user_id, session_id: str):
    temp_files.append(f"{out_path}/coeff##{session_id}.wav")
    temp_files.append(f"{out_path}/{session_id}.wav")
    temp_files.append(f"{out_path}/coeff##{session_id}.txt")
    temp_files.append(f"{out_path}/coeff##{session_id}.mat")
    temp_files.append(f"{out_path}/coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_{user_id}##{session_id}.mp4")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.txt")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.mp4")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.mat")


@app.post("/presave-photo")
async def upload_photo(user_id: str = Form(...), image: UploadFile = File(...)):
    os.makedirs("/app/user_img", exist_ok=True)
    img_path = f"/app/user_img/{user_id}.jpg"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    return {"message": "Photo uploaded successfully", "user_id": user_id}


@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
