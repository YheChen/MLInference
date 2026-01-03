from fastapi import FastAPI

from app.api.routes import router as api_router
from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.metrics.prometheus import metrics_router
from app.middleware.backpressure import BackpressureMiddleware
from app.middleware.timeout import TimeoutMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="ML Inference Service", version="0.1.0")

    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(BackpressureMiddleware)

    # API routes
    app.include_router(api_router)

    # Metrics endpoint
    app.include_router(metrics_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        from app.model.loader import load_model

        model = load_model()
        predictor = Predictor(model)
        batcher = InferenceBatcher(predictor)
        batcher.start()

        app.state.model = model
        app.state.predictor = predictor
        app.state.batcher = batcher

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        batcher = getattr(app.state, "batcher", None)
        if batcher is not None:
            await batcher.stop()

    return app


app = create_app()
