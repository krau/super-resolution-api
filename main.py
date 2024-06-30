import pathlib
import tempfile

import httpx
from fastapi import (
    BackgroundTasks,
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

from config import settings
from sr import process_image


async def verify_token(x_token: str = Header()):
    if x_token != settings.get("token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Token"
        )


app = FastAPI(dependencies=[Depends(verify_token)])


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
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    tile_size: int = Form(default=64),
    scale: int = Form(default=4),
    skip_alpha: bool = Form(default=False),
    resize_to: str | None = Form(default=None),
    url: str | None = Form(default=None),
):
    if (file or url) is None:
        return {"message": "No file or url provided"}
    with tempfile.NamedTemporaryFile(dir="./temp", delete=False) as temp:
        if url is not None:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return {"message": "Failed to download the image"}
                if response.headers.get("Content-Type") not in [
                    "image/jpeg",
                    "image/png",
                ]:
                    return {"message": "Invalid image format"}
                temp.write(response.content)
                temp_path = pathlib.Path(temp.name)
        else:
            temp.write(file.file.read())
            temp_path = pathlib.Path(temp.name)
        background_tasks.add_task(
            process_image,
            tile_size=tile_size,
            input_image=temp_path,
            scale=scale,
            skip_alpha=skip_alpha,
            resize_to=resize_to,
        )
    return {"message": "Success"}


if __name__ == "__main__":
    if not pathlib.Path("temp").exists():
        pathlib.Path("temp").mkdir()
    import uvicorn

    try:
        uvicorn.run(
            app,
            host=settings.get("host", "0.0.0.0"),
            port=settings.get("port", 39721),
        )
    except KeyboardInterrupt:
        print("Server stopped.")
