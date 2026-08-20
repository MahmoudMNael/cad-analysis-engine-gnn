from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.exception_handlers import install_exception_handlers
from apps.api.routes import admin, feedback, health, jobs, models
from packages.common.logging import configure_logging
from packages.store.db.migrate import migrate


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="GNN CAD Analyzer",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
    app.include_router(admin.router, prefix="/api/v1", tags=["admin"])

    @app.on_event("startup")
    def _startup() -> None:
        try:
            migrate()
        except Exception:
            # Health endpoint reports dependency failures; OpenAPI export must still work.
            pass

    return app
