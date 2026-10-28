from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nace_sdk import Choice, NaceClient, Noul, Score

VERSION = "0.1.0"
CLIENT_ID = f"crewai-nace/{VERSION}"
DEFAULT_MODEL = "drex-v2.0"
IMAGE_MODEL = "drex-max"
MAX_QUESTIONS = {"drex-max": 16, "drex-v2.0": 512}
MAX_IMAGES = 8
QUESTION_TYPES = {"noul": Noul, "choice": Choice, "score": Score}


class DecideError(Exception):
    """The check was not sent. This is not a failed guardrail."""


def open_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NaceClient:
    return NaceClient(api_key=api_key, base_url=base_url, client_info=CLIENT_ID)


def _model(model: str | None) -> str:
    if model is None:
        return DEFAULT_MODEL
    if not isinstance(model, str) or not model.strip():
        raise DecideError("model must be a non-empty string")
    return model


def _as_question(question: Any) -> Noul | Choice | Score:
    if isinstance(question, str):
        raise DecideError("a free-form string is not a Drex question")
    if isinstance(question, (Noul, Choice, Score)):
        return question
    if isinstance(question, Mapping):
        kind = question.get("type")
        cls = QUESTION_TYPES.get(kind) if isinstance(kind, str) else None
        if cls is None:
            raise DecideError("question type must be noul, choice, or score")
        try:
            return cls.model_validate(dict(question))
        except Exception as exc:
            raise DecideError("question dict did not match a Drex question") from exc
    raise DecideError("question must be a nace_sdk question or a question dict")


def _check_batch(
    questions: Mapping[str, Any],
    model: str,
    images: Sequence[Any] | None,
) -> dict[str, Noul | Choice | Score]:
    if not isinstance(questions, Mapping) or not questions:
        raise DecideError("questions must be a non-empty mapping")
    built = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise DecideError("question names must be non-empty strings")
        built[name] = _as_question(question)
    cap = MAX_QUESTIONS.get(model)
    if cap is not None and len(built) > cap:
        raise DecideError(f"{model} accepts at most {cap} questions")
    if images:
        if model != IMAGE_MODEL:
            raise DecideError("images require model='drex-max'")
        if len(images) > MAX_IMAGES:
            raise DecideError(f"{IMAGE_MODEL} accepts at most {MAX_IMAGES} images")
    return built


def _state_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    raw = getattr(output, "raw", None)
    if isinstance(raw, str):
        return raw
    raise DecideError("task output must be a string or have a string .raw")


def _client_call(client: Any, owned: bool, call: Callable[[], Any]) -> Any:
    try:
        return call()
    finally:
        if owned:
            client.close()


def _probability(answer: Any) -> str:
    if getattr(answer, "type", None) == "noul":
        return f"{float(answer.noul):.4f}"
    confidence = getattr(answer, "confidence", None)
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        return f"{float(confidence):.4f}"
    return "unknown"


class DrexGuardrail:
    """CrewAI task guardrail. Return (False, feedback) so the task retries."""

    def __init__(
        self,
        questions: Mapping[str, Any],
        *,
        fail_if: Callable[[str, Any], bool],
        model: str | None = None,
        images: Sequence[Any] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not callable(fail_if):
            raise DecideError("fail_if must be a callable")
        self.model = _model(model)
        self.questions = _check_batch(questions, self.model, images)
        self.fail_if = fail_if
        self.images = list(images) if images else None
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    def __call__(self, output: Any) -> tuple[bool, Any]:
        state = _state_text(output)
        owned = self._client is None
        client = self._client or open_client(api_key=self._api_key, base_url=self._base_url)

        def call() -> Any:
            return client.system_one(
                state,
                self.questions,
                model=self.model,
                images=self.images,
            )

        response = _client_call(client, owned, call)
        missing = [name for name in self.questions if name not in response.answers]
        if missing:
            raise DecideError("missing answers: " + ", ".join(missing))
        failed = [
            (name, response.answers[name])
            for name in self.questions
            if self.fail_if(name, response.answers[name])
        ]
        if not failed:
            return True, output
        lines = [
            f"{name} failed (probability {_probability(answer)})"
            for name, answer in failed
        ]
        return False, "\n".join(lines)


def drex_router(
    options: Mapping[str, str | None],
    *,
    threshold: float,
    fallback: str | None = None,
    instructions: str | None = None,
    model: str | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable[[Any], str | None]:
    """Pick a Flow route. Below `threshold`, return `fallback` for the caller's LLM router."""
    if not isinstance(options, Mapping) or len(options) < 2:
        raise DecideError("a router needs at least two options")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise DecideError("threshold must be a number")
    chosen = _model(model)
    question = Choice(instructions=instructions, criteria=dict(options))
    _check_batch({"route": question}, chosen, None)

    def route(state: Any) -> str | None:
        text = _state_text(state)
        owned = client is None
        caller = client or open_client(api_key=api_key, base_url=base_url)

        def call() -> Any:
            return caller.system_one(text, {"route": question}, model=chosen)

        response = _client_call(caller, owned, call)
        answer = response.answers.get("route")
        if answer is None or getattr(answer, "type", None) != "choice":
            raise DecideError("router did not get a choice answer")
        if answer.confidence < threshold:
            return fallback
        if answer.choice not in options:
            raise DecideError(f"unexpected route {answer.choice}")
        return answer.choice

    return route


class DrexDecideTool:
    """Tool body for a CrewAI agent. `as_base_tool` wraps it when crewai is installed."""

    name = "drex_decide"
    description = (
        "Ask Drex one typed question. question_json is one noul, choice, or score object. "
        "Free-form text is rejected."
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        images: Sequence[Any] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = _model(model)
        self.images = list(images) if images else None
        _check_batch({"q": Noul(instructions="check")}, self.model, self.images)
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    def _run(self, state: str, question_json: str) -> str:
        if not isinstance(state, str) or not state:
            raise DecideError("state must be a non-empty string")
        try:
            raw = json.loads(question_json)
        except json.JSONDecodeError as exc:
            raise DecideError("question_json must be JSON") from exc
        built = _check_batch({"q": _as_question(raw)}, self.model, self.images)
        owned = self._client is None
        caller = self._client or open_client(api_key=self._api_key, base_url=self._base_url)

        def call() -> Any:
            return caller.system_one(
                state,
                built,
                model=self.model,
                images=self.images,
            )

        response = _client_call(caller, owned, call)
        answer = response.answers.get("q")
        if answer is None:
            raise DecideError("missing answer")
        return json.dumps(answer.model_dump())


def as_base_tool(tool: DrexDecideTool) -> Any:
    from crewai.tools import BaseTool

    class CrewDrexDecideTool(BaseTool):
        name: str = tool.name
        description: str = tool.description

        def _run(self, state: str, question_json: str) -> str:
            return tool._run(state, question_json)

    return CrewDrexDecideTool()
