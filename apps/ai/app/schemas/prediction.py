from pydantic import BaseModel, Field
from typing import Optional


class VideoMetadata(BaseModel):
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class ViolationResult(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    verdict: str
    violations: list[ViolationResult] = Field(default_factory=list)
    model_name: Optional[str] = None