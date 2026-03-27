"""
RAG-Anything FastAPI service.

Exposes RAG-Anything (multimodal RAG) as HTTP API: query, process documents, insert content.
Multi-tenant: pass workspace (tenant id) on every request; data is isolated by LightRAG workspace.
Configure via .env (see env.example). Use `uv run` to run with server deps.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn

# Add project root for raganything import when running as script
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Config loads .env and exposes get_lightrag_kwargs(workspace)
from app.config import (
    WORKING_DIR,
    OUTPUT_DIR,
    PARSE_METHOD,
    PARSER,
    ENABLE_IMAGE_PROCESSING,
    ENABLE_TABLE_PROCESSING,
    ENABLE_EQUATION_PROCESSING,
    WORKSPACE_DEFAULT,
    get_lightrag_kwargs,
    DOC_PROCESSING_BASE_URL,
    DOC_PROCESSING_LLM_PROVIDER,
    DOC_PROCESSING_EMBEDDING_PROVIDER,
    LLM_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    LIGHTRAG_KV_STORAGE,
    LIGHTRAG_VECTOR_STORAGE,
    LIGHTRAG_GRAPH_STORAGE,
    LIGHTRAG_DOC_STATUS_STORAGE,
)
from app import db_config
from app.llm_client import DocProcessingLLMClient
from app.workspace_config import WorkspaceConfigPayload, merge_workspace_config


# Per-workspace RAG instance cache (workspace -> RAGAnything); config source of truth: DB table ra_litelag_config
_rag_cache: dict[str, "RAGAnything"] = {}
_CONFIG_KEYS = {"working_dir", "parser", "parse_method", "enable_image_processing", "enable_table_processing", "enable_equation_processing"}


def _get_rag(workspace: str, db_overrides: dict | None = None) -> "RAGAnything":
    """Build RAGAnything for workspace. db_overrides from DB are merged over env defaults."""
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    if not DOC_PROCESSING_BASE_URL:
        raise ValueError(
            "Set DOC_PROCESSING_BASE_URL for LLM completions via doc-processing."
        )

    lightrag_kwargs = get_lightrag_kwargs(workspace)
    working_dir = WORKING_DIR
    parser = PARSER
    parse_method = PARSE_METHOD
    enable_image = ENABLE_IMAGE_PROCESSING
    enable_table = ENABLE_TABLE_PROCESSING
    enable_equation = ENABLE_EQUATION_PROCESSING
    if db_overrides:
        for k, v in db_overrides.items():
            if k in lightrag_kwargs:
                lightrag_kwargs[k] = v
            if k in _CONFIG_KEYS:
                if k == "working_dir":
                    working_dir = v
                elif k == "parser":
                    parser = v
                elif k == "parse_method":
                    parse_method = v
                elif k == "enable_image_processing":
                    enable_image = v
                elif k == "enable_table_processing":
                    enable_table = v
                elif k == "enable_equation_processing":
                    enable_equation = v
    lightrag_kwargs["working_dir"] = working_dir
    lightrag_kwargs["workspace"] = workspace

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
        enable_image_processing=enable_image,
        enable_table_processing=enable_table,
        enable_equation_processing=enable_equation,
    )

    llm_client = DocProcessingLLMClient(
        base_url=DOC_PROCESSING_BASE_URL,
        provider=DOC_PROCESSING_LLM_PROVIDER,
        model=LLM_MODEL,
    )
    embedding_client = DocProcessingLLMClient(
        base_url=DOC_PROCESSING_BASE_URL,
        provider=DOC_PROCESSING_EMBEDDING_PROVIDER,
        model=EMBEDDING_MODEL,
    )

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            model=kwargs.pop("model", None),
            reasoning_effort=kwargs.pop("reasoning_effort", None),
            response_format=kwargs.pop("response_format", None),
            **kwargs,
        )

    async def vision_model_func(
        prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs
    ):
        if messages:
            return await llm_client.complete(
                messages=messages,
                model=kwargs.pop("model", None),
                reasoning_effort=kwargs.pop("reasoning_effort", None),
                response_format=kwargs.pop("response_format", None),
                **kwargs,
            )
        if image_data:
            vision_messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": f"{prompt}\n\n[image_base64]\n{image_data}",
                },
            ]
            return await llm_client.complete(
                messages=[
                    m for m in vision_messages if m is not None
                ],
                model=kwargs.pop("model", None),
                reasoning_effort=kwargs.pop("reasoning_effort", None),
                response_format=kwargs.pop("response_format", None),
                **kwargs,
            )
        return await llm_model_func(prompt, system_prompt, history_messages or [], **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=lambda texts: embedding_client.embeddings(
            input_texts=texts,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )


async def get_rag(workspace: str) -> "RAGAnything":
    """Return RAG for workspace; load config from DB if present, then build and cache."""
    if workspace in _rag_cache:
        return _rag_cache[workspace]
    db_overrides = await db_config.get_config(workspace) if db_config._pool else None
    _rag_cache[workspace] = _get_rag(workspace, db_overrides)
    return _rag_cache[workspace]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init Postgres pool, ensure table, load workspace configs from DB into _rag_cache. Shutdown: close pool, finalize RAGs."""
    try:
        await db_config.init_pool()
        for workspace_id in await db_config.list_workspace_ids():
            try:
                await get_rag(workspace_id)
            except Exception:
                pass
    except Exception:
        pass
    yield
    for rag in list(_rag_cache.values()):
        try:
            await rag.finalize_storages()
        except Exception:
            pass
    _rag_cache.clear()
    await db_config.close_pool()


app = FastAPI(
    title="RAG-Anything API",
    description="Multimodal RAG service: query, process documents, insert content. Multi-tenant via workspace parameter.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness and readiness probes"},
        {"name": "config", "description": "App and per-workspace config (GET/POST)"},
        {"name": "query", "description": "Text and multimodal RAG queries"},
        {"name": "ingest", "description": "Content insert and document processing"},
    ],
    servers=[{"url": "/", "description": "Default (relative to base URL)"}],
)


# --- Request/Response models ---

class QueryRequest(BaseModel):
    workspace: str = Field(default=WORKSPACE_DEFAULT, description="Tenant/workspace id for data isolation")
    query: str = Field(..., description="Natural language question")
    mode: str = Field("hybrid", description="Query mode: local, global, hybrid, naive, mix, bypass")
    vlm_enhanced: bool | None = Field(None, description="Use VLM for images in context")
    system_prompt: str | None = None


class QueryResponse(BaseModel):
    answer: str


class MultimodalItem(BaseModel):
    type: str = Field(..., description="One of: table, equation, image")
    table_data: str | None = None
    table_body: str | None = None
    table_caption: str | list[str] | None = None
    table_footnote: str | list[str] | None = None
    latex: str | None = None
    equation_caption: str | None = None
    img_path: str | None = None
    image_caption: str | list[str] | None = None
    image_footnote: str | list[str] | None = None


class QueryMultimodalRequest(BaseModel):
    workspace: str = Field(default=WORKSPACE_DEFAULT, description="Tenant/workspace id")
    query: str = Field(..., description="Question in context of the given multimodal content")
    multimodal_content: list[MultimodalItem] = Field(..., description="Tables, equations, or image refs")
    mode: str = Field("hybrid", description="Query mode")


class ContentListItem(BaseModel):
    type: str = Field(..., description="text, image, table, equation, or custom")
    text: str | None = None
    img_path: str | None = None
    image_caption: list[str] | None = None
    image_footnote: list[str] | None = None
    table_body: str | None = None
    table_caption: list[str] | None = None
    table_footnote: list[str] | None = None
    latex: str | None = None
    content: str | None = None
    page_idx: int = 0


class InsertContentRequest(BaseModel):
    workspace: str = Field(default=WORKSPACE_DEFAULT, description="Tenant/workspace id")
    content_list: list[ContentListItem] = Field(..., description="Pre-parsed content items")
    file_path: str = Field("unknown_document", description="Reference name for citations")
    doc_id: str | None = None
    split_by_character: str | None = None
    split_by_character_only: bool = False


# --- Endpoints ---

@app.get("/health", response_class=PlainTextResponse, tags=["health"])
async def health():
    """Liveness: always 200. Use /ready for readiness."""
    return "ok"


@app.get("/ready", response_class=PlainTextResponse, tags=["health"])
async def ready():
    """Readiness: 200 if doc-processing endpoint is configured."""
    if not DOC_PROCESSING_BASE_URL:
        raise HTTPException(status_code=503, detail="DOC_PROCESSING_BASE_URL not set")
    return "ready"


@app.get("/config", tags=["config"])
async def config_endpoint():
    """Return non-secret app config (storage types, default workspace, parser)."""
    return {
        "working_dir": WORKING_DIR,
        "workspace_default": WORKSPACE_DEFAULT,
        "parser": PARSER,
        "parse_method": PARSE_METHOD,
        "doc_processing_base_url": DOC_PROCESSING_BASE_URL,
        "doc_processing_llm_provider": DOC_PROCESSING_LLM_PROVIDER,
        "lightrag_kv_storage": LIGHTRAG_KV_STORAGE,
        "lightrag_vector_storage": LIGHTRAG_VECTOR_STORAGE,
        "lightrag_graph_storage": LIGHTRAG_GRAPH_STORAGE,
        "lightrag_doc_status_storage": LIGHTRAG_DOC_STATUS_STORAGE,
    }


@app.get("/config/{workspace_id}", tags=["config"])
async def get_workspace_config(workspace_id: str):
    """Get config stored for the given workspace (from DB). 404 if not set."""
    cfg = await db_config.get_config(workspace_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"No config for workspace {workspace_id!r}")
    return cfg


@app.post("/config/{workspace_id}", tags=["config"])
async def set_workspace_config(workspace_id: str, config: WorkspaceConfigPayload):
    """
    Set config for workspace. Payload is partial; missing keys are filled from env defaults.
    Only known keys are allowed (validation error for unknown keys). Stored in DB and _rag_cache updated.
    """
    if db_config._pool is None:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL not configured; set POSTGRES_HOST, POSTGRES_DATABASE, etc.",
        )
    payload = config.model_dump(exclude_none=True)
    full_config = merge_workspace_config(workspace_id, payload)
    await db_config.set_config(workspace_id, full_config)
    if workspace_id in _rag_cache:
        del _rag_cache[workspace_id]
    await get_rag(workspace_id)
    return {"status": "ok", "workspace_id": workspace_id}


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(req: QueryRequest):
    """Text query over the indexed knowledge base for the given workspace."""
    rag = await get_rag(req.workspace)
    await rag._ensure_lightrag_initialized()
    kwargs = {}
    if req.system_prompt is not None:
        kwargs["system_prompt"] = req.system_prompt
    if req.vlm_enhanced is not None:
        kwargs["vlm_enhanced"] = req.vlm_enhanced
    answer = await rag.aquery(req.query, mode=req.mode, **kwargs)
    return QueryResponse(answer=answer)


@app.post("/query/multimodal", response_model=QueryResponse, tags=["query"])
async def query_multimodal(req: QueryMultimodalRequest):
    """Query with inline multimodal content (tables, equations, image paths)."""
    rag = await get_rag(req.workspace)
    await rag._ensure_lightrag_initialized()
    raw = [item.model_dump(exclude_none=True) for item in req.multimodal_content]
    answer = await rag.aquery_with_multimodal(req.query, multimodal_content=raw, mode=req.mode)
    return QueryResponse(answer=answer)


@app.post("/content/insert", tags=["ingest"])
async def insert_content(req: InsertContentRequest):
    """Insert a pre-parsed content list (no file parsing) into the given workspace."""
    rag = await get_rag(req.workspace)
    await rag._ensure_lightrag_initialized()
    raw = [item.model_dump(exclude_none=True) for item in req.content_list]
    await rag.insert_content_list(
        content_list=raw,
        file_path=req.file_path,
        doc_id=req.doc_id,
        split_by_character=req.split_by_character,
        split_by_character_only=req.split_by_character_only,
    )
    return {"status": "ok", "file_path": req.file_path, "workspace": req.workspace}


@app.post("/documents/process", tags=["ingest"])
async def process_document(
    file: UploadFile = File(...),
    workspace: str = Form(default=WORKSPACE_DEFAULT, description="Tenant/workspace id"),
    output_dir: str | None = Form(None),
    parse_method: str | None = Form(None),
    parser: str | None = Form(None),
    doc_id: str | None = Form(None),
):
    """Upload a document and run full RAG processing (parse + index) for the given workspace."""
    rag = await get_rag(workspace)
    await rag._ensure_lightrag_initialized()
    output_dir = output_dir or OUTPUT_DIR
    parse_method = parse_method or PARSE_METHOD
    os.makedirs(output_dir, exist_ok=True)
    if parser:
        from raganything.parser import get_parser
        rag.update_config(parser=parser)
        rag.doc_parser = get_parser(parser)

    suffix = Path(file.filename or "doc").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Upload read failed: {e}")

    try:
        await rag.process_document_complete(
            file_path=tmp_path,
            output_dir=output_dir,
            parse_method=parse_method,
            doc_id=doc_id,
            file_name=file.filename,
        )
        return {"status": "ok", "filename": file.filename, "output_dir": output_dir, "workspace": workspace}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
