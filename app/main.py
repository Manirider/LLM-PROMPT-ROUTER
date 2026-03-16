import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.classifier import classify_intent, strip_override_prefix
from app.exceptions import (
    ClassificationError,
    EmptyMessageError,
    LLMAPIError,
    PromptRouterError,
    RoutingError,
)
from app.logger import log_route_decision
from app.models import HealthResponse, RouteRequest, RouteResponse
from app.router import route_and_respond
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="LLM Prompt Router",
    description="Production-grade AI service that classifies user intent and routes messages to specialized expert personas for high-quality responses.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
logger.info(f"USE_OLLAMA loaded from settings: {settings.use_ollama}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EmptyMessageError)
async def empty_message_handler(_request: Request, exc: EmptyMessageError):
    return JSONResponse(status_code=422, content={"error": exc.message})


@app.exception_handler(ClassificationError)
async def classification_error_handler(_request: Request, exc: ClassificationError):
    return JSONResponse(status_code=502, content={"error": exc.message})


@app.exception_handler(RoutingError)
async def routing_error_handler(_request: Request, exc: RoutingError):
    return JSONResponse(status_code=502, content={"error": exc.message})


@app.exception_handler(LLMAPIError)
async def llm_api_error_handler(_request: Request, exc: LLMAPIError):
    return JSONResponse(status_code=503, content={"error": exc.message})


@app.exception_handler(PromptRouterError)
async def generic_prompt_router_handler(_request: Request, exc: PromptRouterError):
    return JSONResponse(status_code=500, content={"error": exc.message})


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred. Please try again later."},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse()


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
async def route_message(request: RouteRequest):
    start_time = time.perf_counter()
    message = request.message.strip()
    if not message:
        raise EmptyMessageError()
    logger.info("Incoming request: '%s' (%d chars)", message[:80], len(message))
    intent_result = await classify_intent(message)
    logger.info(
        "Classified: intent=%s, confidence=%.2f",
        intent_result.intent.value,
        intent_result.confidence,
    )
    clean_message = strip_override_prefix(message)
    response_text = await route_and_respond(clean_message, intent_result)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    routing_method = (
        "manual"
        if intent_result.confidence == 1.0 and message.startswith("@")
        else "auto"
    )
    if intent_result.confidence < settings.confidence_threshold:
        routing_method = "fallback"
    await log_route_decision(
        intent=intent_result.intent.value,
        confidence=intent_result.confidence,
        user_message=message,
        final_response=response_text,
        routing_method=routing_method,
        latency_ms=elapsed_ms,
    )
    logger.info("Request completed in %.1fms (method=%s)", elapsed_ms, routing_method)
    return RouteResponse(intent=intent_result, response=response_text)
