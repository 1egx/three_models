from runners.BUSI_runner import run_breast
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

import uuid
import base64
import os
import shutil


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/clear_files")
async def clear_files():
    """清理上传和输出的文件"""
    try:
        # 清理uploads目录
        if os.path.exists("uploads"):
            for file in os.listdir("uploads"):
                file_path = os.path.join("uploads", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # 清理outputs目录
        if os.path.exists("outputs"):
            for file in os.listdir("outputs"):
                file_path = os.path.join("outputs", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return {"message": "文件清理完成"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/infer")
async def infer(file: UploadFile = File(...), model: str = Form(...)):
    try:
        import uuid, os

        file_id = str(uuid.uuid4())
        input_path = f"uploads/{file_id}.jpg"
        output_path = f"outputs/{file_id}.jpg"

        with open(input_path, "wb") as f:
            f.write(await file.read())


        if model == "breast":
            run_breast(input_path, output_path)
        with open(output_path, "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        print("🔥 真实报错:", e)
        return {"error": str(e)}