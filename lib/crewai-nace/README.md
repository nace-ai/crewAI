# crewai-nace

CrewAI hooks for Drex. Calls go through `nace-sdk` 0.4.0 (`NaceClient.system_one`). This package does not send HTTP itself.

Omit `model` and the call uses `drex-v2.0`, not `drex-latest`. Any other non-empty id, including `drex-latest`, is forwarded. Images require `model="drex-max"` and are rejected before the call otherwise. `drex-max` rejects more than 16 questions before the call, and `drex-v2.0` rejects more than 512. An unknown id has no client-side question cap. A free-form string is not a question. Omitted instructions are sent as `""`. When this package creates the client, the HTTP timeout is 60 seconds, the retry budget is 180 seconds, and `X-Nace-Client` is `crewai-nace/0.1.0`. The SDK overwrites `User-Agent`, so this package does not set one.

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
