import pathlib
import pickle
import shutil
import tempfile
import threading
import time

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
from logger import logger

import common
from config import settings
from sr_queue import listen_distributed_queue, listen_queue


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


def register_routes():
    @app.get("/")
    async def root():
        return {
            "message": f"Super Resolution API is running as {settings.get('mode', 'single')} mode"
        }

    @app.get("/result/{task_id}")
    async def get_result(task_id: str):
        result = common.redis_client.get(f"{common.RESULT_KEY_PREFIX}{task_id}")
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
            )
        result_data: dict[str, str] = pickle.loads(result)
        return {"result": result_data}

    @app.get("/result/{task_id}/download")
    async def download_result(task_id: str):
        result = common.redis_client.get(f"{common.RESULT_KEY_PREFIX}{task_id}")
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


def register_single_sr_route():
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
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file or url provided",
            )
        temp = tempfile.NamedTemporaryFile(
            dir=settings.get("temp_dir", "./temp"), delete=False
        )
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
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the image",
            )
        resp = common.redis_client.xadd(
            common.BASE_STREAM_NAME,
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
        xlength = common.redis_client.xlen(common.BASE_STREAM_NAME)
        if xlength > 1:
            common.redis_client.set(
                f"{common.RESULT_KEY_PREFIX}{resp.decode('utf-8')}",
                pickle.dumps({"status": "pending"}),
                ex=86400,
            )
        logger.info(f"Task added to queue: {resp.decode('utf-8')}")
        return {"message": "Success", "task_id": f"{resp.decode('utf-8')}"}


def register_master():
    @app.post("/register")
    async def register_worker(
        worker_id: str = Form(...),
        worker_url: str = Form(...),
        worker_token: str = Form(...),
    ):
        try:
            common.redis_client.set(
                f"{common.WORKER_KEY_PREFIX}{worker_id}",
                f"{worker_url}|{worker_token}",
                ex=settings.get("worker_expire", 120),
            )
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register worker",
            )
        return {"message": "Success"}

    @app.get("/workers")
    async def get_workers():
        workers = common.redis_client.keys(f"{common.WORKER_KEY_PREFIX}*")
        return {
            "workers": [
                {
                    "id": worker.decode("utf-8").split("_")[-1],
                    "data": common.redis_client.get(worker).decode("utf-8"),
                }
                for worker in workers
            ]
        }

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
        """
        将输入图片分块, 分发给存储在 Redis 中的 worker

        对于客户端来说, 该 /sr 路由和 single 模式是兼容的
        """
        if (file or url) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file or url provided",
            )

        workers = common.redis_client.keys(f"{common.WORKER_KEY_PREFIX}*")
        if not workers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No available worker",
            )

        input_temp = tempfile.NamedTemporaryFile(
            dir=settings.get("temp_dir", "./temp"), delete=False
        )
        input_path = pathlib.Path(input_temp.name)
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
                    input_temp.write(response.content)
            else:
                if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid image format",
                    )
                input_temp.write(file.file.read())
        except Exception as e:
            logger.error(f"process image error: {e}")
            input_temp.close()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the image",
            )

        workers = common.redis_client.keys(f"{common.WORKER_KEY_PREFIX}*")
        if not workers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No available worker",
            )
        try:
            save_dir = pathlib.Path(
                f"{settings.get('output_dir','./output')}/{input_temp.name.split('/')[-1]}"
            )
            origin_width, origin_height = common.get_image_size(input_path)
            origin_tiles_info = common.split_image(
                input_path,
                save_dir,
                common.calculate_grid(origin_width, origin_height, len(workers)),
            )

            response = {}

            for index, worker_key in enumerate(workers):
                worker = common.redis_client.get(worker_key)
                worker_url, token = worker.decode("utf-8").split("|")
                tile_info = origin_tiles_info[index]

                with open(tile_info.filpath, "rb") as tile_file:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            worker_url + "/", headers={"X-Token": token}
                        )
                        if resp.status_code != 200:
                            raise Exception(f"Worker {worker_url} is not available")
                        resp = await client.post(
                            url=f"{worker_url}/sr",
                            files={"file": tile_file},
                            data={
                                "tile_size": tile_size,
                                "scale": scale,
                                "skip_alpha": skip_alpha,
                                "resize_to": resize_to,
                                "timeout": timeout,
                                "model": model,
                            },
                            headers={"X-Token": token},
                        )
                if resp.status_code != 200:
                    raise Exception(
                        f"Woker {worker_url} failed to process the image: {resp.text}"
                    )

                resp_dict = resp.json().copy()
                resp_dict["tile_info"] = tile_info
                response[worker_key] = resp_dict

        except Exception as e:
            logger.error(f"error: {e}")
            input_temp.close()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process the image: {e}",
            )

        resp = common.redis_client.xadd(
            common.DISTRIBUTED_STREAM_NAME,
            {
                "data": pickle.dumps(
                    {
                        "input_image": input_path,
                        "worker_response": response,
                        "scale": scale,
                    }
                )
            },
        )
        common.redis_client.set(
            f"{common.RESULT_KEY_PREFIX}{resp.decode('utf-8')}",
            pickle.dumps({"status": "pending"}),
            ex=86400,
        )

        return {"message": "Success", "task_id": f"{resp.decode('utf-8')}"}


def register_slave():
    register_single_sr_route()

    def register():
        while True:
            try:
                with httpx.Client() as client:
                    resp = client.post(
                        url=f"{settings.get('master_url')}/register",
                        data={
                            "worker_id": settings.get("worker_id"),
                            "worker_url": settings.get("worker_url"),
                            "worker_token": settings.get("token"),
                        },
                        headers={"X-Token": settings.get("master_token")},
                    )
                    if resp.status_code != 200:
                        logger.error(f"Failed to register to master: {resp.text}")
            except Exception as e:
                logger.error(f"Registration error: {e}")
            finally:
                time.sleep(settings.get("register_interval", 30))

    register_thread = threading.Thread(target=register)
    register_thread.daemon = True
    register_thread.start()


if __name__ == "__main__":
    register_routes()
    if settings.get("mode", "single") == "single":
        register_single_sr_route()
    elif settings.get("mode") == "master":
        register_master()
        queue_thread = threading.Thread(target=listen_distributed_queue)
        queue_thread.daemon = True
        queue_thread.start()
    else:
        register_slave()

    if settings.get("mode") != "master":
        queue_thread = threading.Thread(target=listen_queue)
        queue_thread.daemon = True
        queue_thread.start()

    if not pathlib.Path(settings.get("temp_dir", "./temp")).exists():
        pathlib.Path(settings.get("temp_dir", "./temp")).mkdir(parents=True)
    import uvicorn

    try:
        uvicorn.run(
            app,
            host=settings.get("host", "0.0.0.0"),
            port=settings.get("port", 39721),
        )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down")
        common.redis_client.delete(common.BASE_STREAM_NAME)
        if settings.get("mode") == "master":
            common.redis_client.delete(common.DISTRIBUTED_STREAM_NAME)
        shutil.rmtree(settings.get("temp_dir", "./temp"))
