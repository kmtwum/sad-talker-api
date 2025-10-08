from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger
import tempfile
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

facerender_batch_size = 28  # Reduced to prevent OOM
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


def generate_tts(text: str, tts_preference: str = "coqui") -> str:
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url, json={"text": text})
        tts_response.raise_for_status()
        audio_path = f"/tmp/tts_{hash(text)}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("VOICE_ID")

        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
        )

        print("Saving 11 audio file...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"Audio file saved to {f.name}")

        return f.name


@app.post("/generate")
async def predict_image(
        background_tasks: BackgroundTasks,
        text: str = Form(...),
        tts_preference: str = Form("coqui"),
        audio: UploadFile = File(None),
        use_enhancer: bool = False):

    preprocess_mode = "full"
    temp_files = []

    # Create session
    session_id = uuid.uuid4()

    # Save image
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as im:
            im.write(await image.read())
            pic_path = im.name
    else:
        pic_path = "/app/img/avatar.png"

    # Generate TTS
    audio_path = generate_tts(text, tts_preference, out_path, str(session_id))
    temp_files.append(audio_path)

    first_frame_dir = os.path.join(out_path, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, preprocess_mode,
                                                                           source_image_flag=True)
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None
    batch = get_data(first_coeff_path, audio_path, "cuda", ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, out_path, 0, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               facerender_batch_size, None, None, None,
                               expression_scale=1, still_mode=True, preprocess=preprocess_mode)
    video_path = animate_from_coeff.generate_deploy(
        data, out_path, pic_path, crop_info,
        enhancer="gfpgan" if use_enhancer else None,
        background_enhancer=None,
        preprocess=preprocess_mode,
        skip_background_blend=True)

    populate_temp_files(temp_files, out_path, str(session_id))

    # Schedule cleanup after the response is sent
    background_tasks.add_task(cleanup_files, temp_files)

    return FileResponse(video_path, media_type="video/mp4", filename="result.mp4")


def populate_temp_files(temp_files: list, out_path, session_id: str):
    temp_files.append(f"{out_path}/coeff##{session_id}.wav")
    temp_files.append(f"{out_path}/{session_id}.wav")
    temp_files.append(f"{out_path}/coeff##{session_id}.txt")
    temp_files.append(f"{out_path}/coeff##{session_id}.mat")
    temp_files.append(f"{out_path}/coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_coeff##{session_id}.mp4")


@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
