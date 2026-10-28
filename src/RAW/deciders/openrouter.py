import os
from typing import Any, Dict, Mapping, Optional
from pydantic import BaseModel
from RAW.models import Decision, DecisionQuestion, Image, Usage
from RAW.utils import Logger, RequestsClient
from .base import BaseDecider, DecisionState


def _dump(model: BaseModel) -> Dict[str, Any]:
    # Drop unset top-level fields only; None inside values (e.g. a Choice option
    # with no description) is meaningful to the API.
    return {k: v for k, v in model.model_dump().items() if v is not None}


class OpenRouterDeciderOptions(BaseModel):
    # OpenRouter provider routing preferences.
    provider: Optional[Dict[str, Any]] = None
    # Groups related requests for OpenRouter observability; never sent to the model provider.
    session_id: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    user: Optional[str] = None


class OpenRouterDecider(BaseDecider):
    """Decision models through OpenRouter's Decisions API (``POST /api/alpha/decisions``).

    Defaults to TypeSafe's Jev; any decision model on OpenRouter works by changing ``model``.
    Probabilities differ between models, so re-tune thresholds when switching.
    """

    def __init__(
        self,
        model: str = "typesafe/jev-1.14",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api",
        options: Optional[OpenRouterDeciderOptions] = None,
        logger: Optional[Logger] = None,
        client: Optional[RequestsClient] = None
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = client or RequestsClient(
            base_url=base_url,
            headers=headers,
            timeout=300,
            logger=logger
        )
        self.model = model
        self.options = options
        self.logger = logger

    @staticmethod
    def _image_part(image: Image) -> Dict[str, Any]:
        ext = "." + image.mime_type.split("/")[-1]
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image.mime_type};base64,{image.to_base64(ext)}"}
        }

    def _build_request_body(self, state: DecisionState, questions: Mapping[str, DecisionQuestion]) -> Dict[str, Any]:
        if isinstance(state, Image):
            state = [state]
        if isinstance(state, list):
            state = [self._image_part(item) if isinstance(item, Image) else item for item in state]

        body: Dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {name: _dump(question) for name, question in questions.items()},
        }
        if self.options:
            body.update(_dump(self.options))
        return body

    async def decide(self, state: DecisionState, questions: Mapping[str, DecisionQuestion]) -> Decision:
        if not questions:
            raise ValueError("decide() needs at least one question.")

        body = self._build_request_body(state, questions)
        try:
            response = await self.client.post("/alpha/decisions", json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            return Decision(
                answers=data["answers"],
                usage=Usage(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    cost=usage.get("cost"),
                ),
                id=data.get("id"),
                model=data.get("model"),
                provider=data.get("provider"),
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Decision error: {str(e)}")
            raise RuntimeError(f"Decision error: {str(e)}")

    async def stop(self) -> None:
        await self.client.aclose()
