# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from app.routes.health import router as health_router
# from app.routes.video import router as video_router


# app = FastAPI(
#     title="SecureVision AI",
#     description="Local video analysis service for banking compliance",
#     version="0.1.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "http://localhost:4000",
#     ],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(health_router)
# app.include_router(video_router)


# @app.get("/")
# def root():
#     return {
#         "name": "SecureVision AI",
#         "status": "running",
#         "version": "0.1.0",
#     }
from app.routes.full_analysis import (
    router as full_analysis_router,
)
from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.routes.analyze import (
    router as analyze_router,
)

from app.routes.health import (
    router as health_router,
)

from app.routes.video import (
    router as video_router,
)
from app.routes.workplace import (
    router as workplace_router,
)

app = FastAPI(
    title="SecureVision AI",
    description=(
        "Local AI video analysis "
        "service for banking "
        "cash-discipline monitoring"
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4000",
    ],

    allow_credentials=False,

    allow_methods=[
        "*",
    ],

    allow_headers=[
        "*",
    ],
)


app.include_router(
    health_router
)

app.include_router(
    video_router
)

app.include_router(
    analyze_router
)
app.include_router(
    workplace_router
)
app.include_router(
    full_analysis_router
)
@app.get("/")
def root():
    return {
        "name": (
            "SecureVision AI"
        ),

        "status": (
            "running"
        ),

        "version": (
            "1.0.0"
        ),

        "api": {
            "analyze": (
                "POST /api/v1/analyze"
            ),

            "job": (
                "GET /api/v1/jobs/{job_id}"
            ),

            "docs": (
                "/docs"
            ),
        },
    }