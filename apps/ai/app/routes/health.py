from fastapi import APIRouter
import torch
import pytorchvideo

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health():
    return {
        "status": "ok",
        "service": "securevision-ai",
        "pytorch_version": torch.__version__,
        "pytorchvideo_available": True,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }