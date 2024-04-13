from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}