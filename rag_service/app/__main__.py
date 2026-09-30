"""Run the service with host and port from settings: ``python -m app``."""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port)