from fastapi import FastAPI
from app.api.routes import router as api_router
from app.metrics.prometheus import metrics_router


def create_app() -> FastAPI:
    app = FastAPI(title="ML Inference Service", version="0.1.0")

    # API routes
    app.include_router(api_router)

    # Metrics endpoint
    app.include_router(metrics_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        # Later: load model once, start batcher loop
        pass

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        # Later: stop batcher loop gracefully
        pass

    return app


app = create_app()
