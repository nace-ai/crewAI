# crewai-nace

CrewAI hooks for Drex. Calls go through `nace-sdk` 0.2.0 (`NaceClient.system_one`). This package does not send HTTP itself.

Both models are supported. Omit `model` and the call uses `drex-v2.0`, not `drex-latest`. Images require `model="drex-max"` and are rejected before the call otherwise. `drex-max` rejects more than 16 questions before the call. A free-form string is not a question.

## Task guardrail

`DrexGuardrail` is a CrewAI task guardrail. It returns `(True, output)` or `(False, feedback)`. The feedback includes the probability, so CrewAI can retry the task.

```python
from nace_sdk import Noul
from crewai_nace import DrexGuardrail

guard = DrexGuardrail(
    {"policy": Noul(instructions="Does this violate the refund policy?")},
    fail_if=lambda name, answer: answer.noul >= 0.7,
)
task.guardrails = [guard]
```

Pass `images=` only with `model="drex-max"` when the task output includes a chart or screenshot.

## Flow router

`drex_router` picks a route with a `choice` question. If confidence is below `threshold`, it returns `fallback` and does not call an LLM. Point `fallback` at your existing LLM router.

```python
from crewai_nace import drex_router

route = drex_router(
    {"billing": "money questions", "support": "product questions"},
    threshold=0.8,
    fallback="llm_router",
)
```

## Agent tool

`DrexDecideTool._run(state, question_json)` asks one typed question. `question_json` is one `noul`, `choice`, or `score` object. `as_base_tool` wraps it as a CrewAI `BaseTool` when `crewai` is installed.

Not in this package: a docs PR to CrewAI, an example in crewAI-examples, or a newsletter post.
