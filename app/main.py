from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.metrics.prometheus import metrics_router
from app.middleware.backpressure import BackpressureMiddleware
from app.middleware.metrics import MetricsMiddleware
from app.middleware.timeout import TimeoutMiddleware
from app.utils.logging import setup_logging


def create_app() -> FastAPI:
    logger = setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from app.model.loader import load_model

        model = load_model()
        predictor = Predictor(model)
        batcher = InferenceBatcher(predictor)
        batcher.start()

        app.state.model = model
        app.state.predictor = predictor
        app.state.batcher = batcher
        logger.info("Startup complete", extra={
            "detail": "model loaded, batcher started",
        })

        try:
            yield
        finally:
            await batcher.stop()
            logger.info("Shutdown complete")

    app = FastAPI(title="ML Inference Service", version="0.1.0", lifespan=lifespan)

    # Starlette applies the most recently added middleware outermost, so this
    # registration order runs as: metrics -> backpressure -> timeout -> routes.
    # Metrics must be outermost to observe shed and timed-out requests, which
    # never reach the handler.
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(BackpressureMiddleware)
    app.add_middleware(MetricsMiddleware)

    # API routes
    app.include_router(api_router)

    # Metrics endpoint
    app.include_router(metrics_router)

    return app


app = create_app()
