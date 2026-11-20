from fastapi import FastAPI
from app.routes.webhook import router as webhook_router

app = FastAPI(title="LocFacil AI Assistant")

app.include_router(webhook_router)


@app.get("/")
async def root():
    return {"message": "LocFacil AI Assistant backend is running"}