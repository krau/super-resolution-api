import pathlib
import tempfile

from fastapi import FastAPI, File, UploadFile

from sr import process_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Real ESRGAN API is running"}


@app.post("/sr")
async def super_resolution(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(dir="./temp") as temp:
        temp.write(file.file.read())
        temp_path = pathlib.Path(temp.name)
        output_path = process_image(input_image=temp_path)
        return {"output_path": str(output_path)}


if __name__ == "__main__":
    if not pathlib.Path("temp").exists():
        pathlib.Path("temp").mkdir()
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=39721)
    except KeyboardInterrupt:
        print("Server stopped.")
