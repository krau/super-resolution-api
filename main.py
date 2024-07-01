import pathlib
import pickle
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

from common import redis_client
from config import settings
from sr import listen_queue


async def verify_token(x_token: str = Header()):
    if x_token != settings.get("token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Token"
        )


app = FastAPI(
    dependencies=[Depends(verify_token)],
    title="Real ESRGAN API",
    description="Restful API for Real ESRGAN",
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
    return {"message": "Real ESRGAN API is running"}


@app.post("/sr")
async def super_resolution(
    file: UploadFile | None = File(default=None),
    tile_size: int = Form(default=64),
    scale: int = Form(default=4),
    skip_alpha: bool = Form(default=False),
    resize_to: str | None = Form(default=None),
    url: str | None = Form(default=None),
):
    if (file or url) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file or url provided"
        )

    temp = tempfile.NamedTemporaryFile(dir="./temp", delete=False)
    try:
        if url is not None:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return {"message": "Failed to download the image"}
                if response.headers.get("Content-Type") not in [
                    "image/jpeg",
                    "image/png",
                ]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid image format",
                    )
                temp.write(response.content)
                temp_path = pathlib.Path(temp.name)
        else:
            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image format",
                )
            temp.write(file.file.read())
            temp_path = pathlib.Path(temp.name)
    except Exception as e:
        logger.error(f"process image error: {e}")
        temp.close()
    resp = redis_client.xadd(
        "real_esrgan_api_queue",
        {
            "data": pickle.dumps(
                {
                    "input_image": temp_path,
                    "tile_size": tile_size,
                    "scale": scale,
                    "skip_alpha": skip_alpha,
                    "resize_to": resize_to,
                }
            ),
        },
    )
    logger.info(f"Task added to queue: {resp.decode('utf-8')}")
    redis_client.set(
        f"real_esrgan_api_result_{resp.decode('utf-8')}",
        pickle.dumps({"status": "pending"}),
        ex=86400,
    )
    return {"message": "Success", "task_id": f"{resp.decode('utf-8')}"}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = redis_client.get(f"real_esrgan_api_result_{task_id}")
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
        )
    result_data: dict[str, str] = pickle.loads(result)
    return {"result": result_data}


@app.get("/result/{task_id}/download")
async def download_result(task_id: str):
    result = redis_client.get(f"real_esrgan_api_result_{task_id}")
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
        sr_thread.start()
        uvicorn.run(
            app,
            host=settings.get("host", "0.0.0.0"),
            port=settings.get("port", 39721),
        )
    finally:
        redis_client.delete("real_esrgan_api_queue")
        redis_client.close()
        pathlib.Path("temp").rmdir()
