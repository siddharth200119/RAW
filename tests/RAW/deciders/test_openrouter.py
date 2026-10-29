import httpx
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from RAW.deciders import BaseDecider, OpenRouterDecider, OpenRouterDeciderOptions
from RAW.models import ChoiceQuestion, Image, NoulCriteria, NoulQuestion, ScoreQuestion

RESPONSE = {
    "id": "gen-dec-1790015143-AIaTutprXsJ5EwohRSjb",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "is_bug": {"type": "noul", "noul": 0.96},
        "team": {
            "type": "choice",
            "choice": "payments",
            "confidence": 0.67,
            "probabilities": {"payments": 0.78, "frontend": 0.22, "account": 0},
        },
    },
    "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
}

QUESTIONS = {
    "is_bug": NoulQuestion(
        instructions="Is the customer reporting a software defect?",
        criteria=NoulCriteria(true="Broken behaviour.", false="A question or feature request."),
    ),
    "team": ChoiceQuestion(
        instructions="Which team should own this ticket?",
        criteria={"payments": "Checkout issues.", "frontend": "Rendering issues.", "account": None},
    ),
}


def mock_response(data, status_code=200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = data
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    return response


@pytest.fixture
def decider():
    return OpenRouterDecider(api_key="test_key")


def test_is_a_base_decider(decider):
    assert isinstance(decider, BaseDecider)
    assert decider.model == "typesafe/jev-1.14"


def test_auth_header_from_argument_or_env(monkeypatch):
    assert OpenRouterDecider(api_key="abc").client.headers == {"Authorization": "Bearer abc"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert OpenRouterDecider().client.headers == {"Authorization": "Bearer from-env"}

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert OpenRouterDecider().client.headers == {}


async def test_decide_posts_request_and_parses_answers(decider):
    with patch.object(decider.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response(RESPONSE)
        state = {"customer_tier": "enterprise", "ticket": "Checkout shows a blank screen."}
        decision = await decider.decide(state=state, questions=QUESTIONS)

    mock_post.assert_called_once_with(
        "/alpha/decisions",
        json={
            "model": "typesafe/jev-1.14",
            "state": state,
            "questions": {
                "is_bug": {
                    "type": "noul",
                    "instructions": "Is the customer reporting a software defect?",
                    "criteria": {"true": "Broken behaviour.", "false": "A question or feature request."},
                },
                "team": {
                    "type": "choice",
                    "instructions": "Which team should own this ticket?",
                    "criteria": {"payments": "Checkout issues.", "frontend": "Rendering issues.", "account": None},
                },
            },
        },
        is_async=True,
    )
    assert decision.answers["is_bug"].noul == 0.96
    assert decision.answers["team"].choice == "payments"
    assert decision.usage.input_tokens == 476
    assert decision.usage.cost == 0.000019992
    assert decision.model == "typesafe/jev-1.13-20260917"
    assert decision.provider == "TypeSafe"


def test_options_are_merged_into_body():
    decider = OpenRouterDecider(
        model="~typesafe/jev-latest",
        api_key="k",
        options=OpenRouterDeciderOptions(session_id="s-1", provider={"sort": "price"}),
    )
    body = decider._build_request_body("text", {"q": ScoreQuestion(instructions="How?", criteria=["a", "b"])})
    assert body["model"] == "~typesafe/jev-latest"
    assert body["session_id"] == "s-1"
    assert body["provider"] == {"sort": "price"}
    assert "trace" not in body and "user" not in body


def test_images_in_state_become_image_url_parts(decider):
    image = Image(data=np.zeros((4, 4, 3), dtype=np.uint8), mime_type="image/png")
    body = decider._build_request_body(["Find the badge", image], {"q": NoulQuestion(instructions="?")})

    text, part = body["state"]
    assert text == "Find the badge"
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,iVBOR")  # PNG magic bytes

    single = decider._build_request_body(image, {"q": NoulQuestion(instructions="?")})
    assert single["state"][0]["type"] == "image_url"


async def test_empty_questions_rejected(decider):
    with pytest.raises(ValueError):
        await decider.decide(state="x", questions={})


async def test_http_error_raises_runtime_error(decider):
    with patch.object(decider.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response({"error": {"code": 400}}, status_code=400)
        with pytest.raises(RuntimeError, match="Decision error"):
            await decider.decide(state="x", questions=QUESTIONS)


async def test_malformed_response_raises_runtime_error(decider):
    with patch.object(decider.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response({"model": "x", "usage": {}})
        with pytest.raises(RuntimeError, match="Decision error"):
            await decider.decide(state="x", questions=QUESTIONS)


async def test_stop_closes_client(decider):
    with patch.object(decider.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await decider.stop()
        mock_aclose.assert_called_once()
