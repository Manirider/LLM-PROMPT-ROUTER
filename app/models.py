from pydantic import BaseModel, Field
from config.prompts import Intent


class RouteRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message to classify and route.",
        examples=["How do I sort a list in Python?"],
    )


class IntentResult(BaseModel):
    intent: Intent = Field(description="The classified intent category.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the classification (0.0–1.0)."
    )


class RouteResponse(BaseModel):
    intent: IntentResult = Field(description="Classification result.")
    response: str = Field(description="Generated response from the expert persona.")


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    service: str = Field(default="llm-prompt-router")
    version: str = Field(default="1.0.0")
