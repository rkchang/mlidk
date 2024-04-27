from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve the "frontend" directory
app.mount("/", StaticFiles(directory="frontend"), name="frontend")

@app.get("/hello")
async def root():
    return {"message": "Hello World"}