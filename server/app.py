"""
FastAPI server for the Researcher environment.

Run locally:
    uvicorn researcher1.server.app:app --reload --port 8000

Or from inside the researcher1 directory:
    uvicorn server.app:app --reload --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_fastapi_app as create_app

try:
    from researcher1.env.models import ResearchAction, ResearchObservation
    from researcher1.env.researcher_env import ResearcherEnv
except ModuleNotFoundError:
    from env.models import ResearchAction, ResearchObservation
    from env.researcher_env import ResearcherEnv

app = create_app(
    ResearcherEnv,
    ResearchAction,
    ResearchObservation,
    env_name="researcher_env",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
