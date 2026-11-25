import os

from apps.api.app import create_app

app = create_app()


def run() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
