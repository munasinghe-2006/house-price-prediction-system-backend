from fastapi import FastAPI

app = FastAPI(
    title="FastAPI Template",
    description="This is a sample FastAPI template.",
    version="1.0.0",)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Template!"}

@app.get("/health")
def read_health():
    return {"status": "ok"}

