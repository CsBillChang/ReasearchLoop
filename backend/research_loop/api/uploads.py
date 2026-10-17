"""附件入口：接收有界字节流，PDF 身份和文件存储交由上传服务处理。"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.services.uploads import store_pdf

router = APIRouter()


@router.post('/uploads', status_code=201)
async def upload_pdf(
    request: Request,
    filename: str = Query('paper.pdf', max_length=240),
    settings: Settings = Depends(get_runtime_settings),
):
    """接收 PDF 原始流；超限立即停止读取，业务错误转换为 HTTP 422。"""
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > settings.max_pdf_bytes:
            raise HTTPException(413, 'PDF超过允许大小')
    try:
        return store_pdf(settings, bytes(data), filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
