import pathlib
import pickle
import shutil
import tempfile
import threading

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

import common
from config import settings
from sr import listen_queue


async def verify_token(x_token: str = Header()):
    if x_token != settings.get("token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Token"
        )


app = FastAPI(
    dependencies=[Depends(verify_token)],
    title="Super Resolution API",
    description="Super Resolution API for Anime and Illustration",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Super Resolution API is running"}


@app.post("/sr")
async def super_resolution(
    file: UploadFile | None = File(default=None),
    tile_size: int = Form(default=64, ge=32, le=128),
    scale: int = Form(default=4, ge=2, le=8),
    skip_alpha: bool = Form(default=False),
    resize_to: str | None = Form(default=None),
    url: str | None = Form(default=None),
    timeout: int = Form(
        default=common.PROGRESS_TIMEOUT, ge=1, le=common.MAX_ALLOWED_TIMEOUT
    ),
    model: str = Form(default=common.MODEL_NAME_DEFAULT),
):
    if (file or url) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file or url provided"
        )
    temp = tempfile.NamedTemporaryFile(dir="./temp", delete=False)
    temp_path = pathlib.Path(temp.name)
    try:
        if url is not None:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return {"message": "Failed to download the image"}
                if response.headers.get("Content-Type") not in [
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                ]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid image format",
                    )
                temp.write(response.content)
        else:
            if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image format",
                )
            temp.write(file.file.read())
    except Exception as e:
        logger.error(f"process image error: {e}")
        temp.close()
    resp = common.redis_client.xadd(
        common.STREAM_NAME,
        {
            "data": pickle.dumps(
                {
                    "input_image": temp_path,
                    "tile_size": tile_size,
                    "scale": scale,
                    "skip_alpha": skip_alpha,
                    "resize_to": resize_to,
                    "timeout": timeout,
                    "model": model,
                }
            ),
        },
    )
    xlength = common.redis_client.xlen(common.STREAM_NAME)
    if xlength > 1:
        common.redis_client.set(
            f"super_resolution_api_result_{resp.decode('utf-8')}",
            pickle.dumps({"status": "pending"}),
            ex=86400,
        )
    logger.info(f"Task added to queue: {resp.decode('utf-8')}")
    return {"message": "Success", "task_id": f"{resp.decode('utf-8')}"}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = common.redis_client.get(f"super_resolution_api_result_{task_id}")
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
        )
    result_data: dict[str, str] = pickle.loads(result)
    return {"result": result_data}


@app.get("/result/{task_id}/download")
async def download_result(task_id: str):
    result = common.redis_client.get(f"super_resolution_api_result_{task_id}")
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
        )
    result_data: dict[str, str] = pickle.loads(result)
    if result_data["status"] != "success":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is {result_data['status']}",
        )
    file_path = pathlib.Path(result_data["path"])
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        headers={"Content-Length": str(file_path.stat().st_size)},
        media_type="image/png",
    )


if __name__ == "__main__":
    if not pathlib.Path("temp").exists():
        pathlib.Path("temp").mkdir()
    import uvicorn

    try:
        sr_thread = threading.Thread(target=listen_queue)
        sr_thread.daemon = True
        sr_thread.start()
        uvicorn.run(
            app,
            host=settings.get("host", "0.0.0.0"),
            port=settings.get("port", 39721),
        )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down")
        common.redis_client.delete(common.STREAM_NAME)
        shutil.rmtree("temp")
