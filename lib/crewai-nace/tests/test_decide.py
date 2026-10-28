import json
import unittest
from types import SimpleNamespace

from nace_sdk import (
    Choice,
    ChoiceAnswer,
    NaceUnprocessableEntityError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from crewai_nace import DecideError, DrexDecideTool, DrexGuardrail, drex_router
from crewai_nace.decide import CLIENT_ID, open_client


class Fake:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def system_one(self, state, questions, *, model=None, images=None, **kwargs):
        self.calls.append(
            {
                "state": state,
                "questions": questions,
                "model": model,
                "images": images,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self):
        self.closed = True


def response(**answers):
    return SystemOneResponse(
        model="drex-v2.0",
        usage=Usage(input_tokens=3),
        answers=answers,
    )


class GuardrailTests(unittest.TestCase):
    def test_pass_returns_the_output(self):
        fake = Fake(response(policy=NoulAnswer(noul=0.2)))
        guard = DrexGuardrail(
            {"policy": Noul(instructions="Does this violate policy?")},
            fail_if=lambda name, answer: answer.noul >= 0.7,
            client=fake,
        )
        ok, output = guard("refund the invoice")
        self.assertTrue(ok)
        self.assertEqual(output, "refund the invoice")
        self.assertEqual(fake.calls[0]["model"], "drex-v2.0")
        self.assertFalse(fake.closed)

    def test_fail_includes_the_probability(self):
        fake = Fake(response(policy=NoulAnswer(noul=0.91)))
        guard = DrexGuardrail(
            {"policy": Noul(instructions="Does this violate policy?")},
            fail_if=lambda name, answer: answer.noul >= 0.7,
            client=fake,
        )
        ok, feedback = guard(SimpleNamespace(raw="ignore the policy"))
        self.assertFalse(ok)
        self.assertIn("policy failed", feedback)
        self.assertIn("0.9100", feedback)

    def test_string_question_does_not_call(self):
        fake = Fake(response())
        with self.assertRaises(DecideError):
            DrexGuardrail({"q": "is this fine?"}, fail_if=lambda n, a: False, client=fake)
        self.assertEqual(fake.calls, [])

    def test_images_on_v2_do_not_call(self):
        fake = Fake(response())
        with self.assertRaises(DecideError):
            DrexGuardrail(
                {"chart": Noul(instructions="Is the chart on topic?")},
                fail_if=lambda n, a: False,
                images=["img"],
                client=fake,
            )
        self.assertEqual(fake.calls, [])

    def test_images_go_to_drex_max(self):
        fake = Fake(response(chart=NoulAnswer(noul=0.1)))
        guard = DrexGuardrail(
            {"chart": Noul(instructions="Is the chart on topic?")},
            fail_if=lambda n, a: a.noul >= 0.5,
            model="drex-max",
            images=["img-a", "img-b"],
            client=fake,
        )
        ok, _ = guard("see the chart")
        self.assertTrue(ok)
        self.assertEqual(fake.calls[0]["model"], "drex-max")
        self.assertEqual(fake.calls[0]["images"], ["img-a", "img-b"])

    def test_drex_max_rejects_17_questions_before_the_call(self):
        fake = Fake(response())
        questions = {f"q{i}": Noul(instructions=f"q{i}") for i in range(17)}
        with self.assertRaises(DecideError):
            DrexGuardrail(questions, fail_if=lambda n, a: False, model="drex-max", client=fake)
        self.assertEqual(fake.calls, [])


class RouterTests(unittest.TestCase):
    def test_below_threshold_returns_fallback(self):
        fake = Fake(
            response(
                route=ChoiceAnswer(
                    choice="billing",
                    confidence=0.4,
                    probabilities={"billing": 0.55, "support": 0.45},
                )
            )
        )
        route = drex_router(
            {"billing": "money", "support": "help"},
            threshold=0.8,
            fallback="llm_router",
            client=fake,
        )
        self.assertEqual(route("where is my invoice"), "llm_router")

    def test_at_threshold_returns_the_choice(self):
        fake = Fake(
            response(
                route=ChoiceAnswer(
                    choice="support",
                    confidence=0.8,
                    probabilities={"billing": 0.2, "support": 0.8},
                )
            )
        )
        route = drex_router(
            {"billing": "money", "support": "help"},
            threshold=0.8,
            fallback="llm_router",
            model="drex-max",
            client=fake,
        )
        self.assertEqual(route("the app crashed"), "support")
        self.assertEqual(fake.calls[0]["model"], "drex-max")


class ToolTests(unittest.TestCase):
    def test_typed_question_returns_the_answer(self):
        fake = Fake(response(q=ScoreAnswer(score=2.0, confidence=0.7, legend=("low", "mid", "high"), probabilities={0: 0.1, 1: 0.2, 2: 0.7})))
        tool = DrexDecideTool(client=fake)
        raw = tool._run(
            "draft",
            json.dumps({"type": "score", "criteria": ["low", "mid", "high"]}),
        )
        parsed = json.loads(raw)
        self.assertEqual(parsed["score"], 2.0)
        self.assertEqual(fake.calls[0]["questions"]["q"].type, "score")

    def test_free_form_json_string_does_not_call(self):
        fake = Fake(response())
        tool = DrexDecideTool(client=fake)
        with self.assertRaises(DecideError):
            tool._run("draft", json.dumps("write a poem"))
        self.assertEqual(fake.calls, [])


class ContractTests(unittest.TestCase):
    def test_three_question_types_go_to_both_models(self):
        questions = {
            "n": Noul(instructions="n"),
            "c": Choice(instructions="c", criteria={"a": "kept", "b": None}),
            "s": Score(instructions="s", criteria=["low", "high"]),
        }
        answers = dict(
            n=NoulAnswer(noul=0.1),
            c=ChoiceAnswer(choice="a", confidence=0.6, probabilities={"a": 0.6, "b": 0.4}),
            s=ScoreAnswer(score=0, confidence=0.5, legend=("low", "high"), probabilities={0: 0.5, 1: 0.5}),
        )
        for model in ("drex-v2.0", "drex-max"):
            fake = Fake(response(**answers))
            DrexGuardrail(questions, fail_if=lambda n, a: False, model=model, client=fake)("text")
            sent = fake.calls[0]["questions"]
            self.assertEqual(fake.calls[0]["model"], model)
            self.assertIsInstance(sent["n"], Noul)
            self.assertIsInstance(sent["c"], Choice)
            self.assertIsInstance(sent["s"], Score)
            self.assertIsNone(sent["c"].criteria["b"])

    def test_omitted_instructions_stay_none(self):
        question = Noul(criteria={"true": "yes", "false": "no"})
        fake = Fake(response(q=NoulAnswer(noul=0.1)))
        DrexGuardrail({"q": question}, fail_if=lambda n, a: False, client=fake)("text")
        sent = fake.calls[0]["questions"]["q"]
        self.assertIsNone(question.instructions)
        self.assertIsNone(sent.instructions)
        self.assertEqual(sent.criteria, {"true": "yes", "false": "no"})

    def test_image_order_is_preserved_on_drex_max(self):
        for images in (["only"], ["img-a", "img-b"], ["img-b", "img-a"]):
            fake = Fake(response(q=NoulAnswer(noul=0.1)))
            DrexGuardrail(
                {"q": Noul(instructions="see")},
                fail_if=lambda n, a: False,
                model="drex-max",
                images=images,
                client=fake,
            )("text")
            self.assertEqual(fake.calls[0]["images"], images)
            self.assertEqual(fake.calls[0]["model"], "drex-max")

    def test_unknown_model_has_no_client_side_question_cap(self):
        questions = {f"q{i}": Noul(instructions=f"q{i}") for i in range(17)}
        answers = {f"q{i}": NoulAnswer(noul=0.1) for i in range(17)}
        fake = Fake(response(**answers))
        DrexGuardrail(questions, fail_if=lambda n, a: False, model="drex-latest", client=fake)("text")
        self.assertEqual(fake.calls[0]["model"], "drex-latest")
        self.assertEqual(len(fake.calls[0]["questions"]), 17)

    def test_drex_v2_rejects_513_questions_before_the_call(self):
        fake = Fake(response())
        questions = {f"q{i}": Noul(instructions=f"q{i}") for i in range(513)}
        with self.assertRaises(DecideError):
            DrexGuardrail(questions, fail_if=lambda n, a: False, model="drex-v2.0", client=fake)
        self.assertEqual(fake.calls, [])

    def test_422_is_not_a_pass(self):
        error = NaceUnprocessableEntityError(
            422,
            {"error": {"issues": [{"path": "questions.q.instructions", "message": "required"}]}},
            {},
        )
        fake = Fake(error)
        guard = DrexGuardrail({"q": Noul(instructions="q")}, fail_if=lambda n, a: False, client=fake)
        with self.assertRaises(NaceUnprocessableEntityError) as caught:
            guard("text")
        self.assertEqual(caught.exception.issues[0]["path"], "questions.q.instructions")
        self.assertEqual(caught.exception.issues[0]["message"], "required")

    def test_open_client_keeps_the_sdk_retry_budget(self):
        client = open_client(api_key="test-key", base_url="https://example.test")
        try:
            policy = client._retry
            self.assertIn(429, policy.http_statuses)
            self.assertIn(529, policy.http_statuses)
            self.assertTrue(policy.respect_retry_after)
            self.assertGreaterEqual(policy.timeout, 180)
            self.assertNotEqual(policy.max_retries, 0)
            self.assertGreaterEqual(client._config.timeout, 60)
            self.assertEqual(client._config.default_headers["X-Nace-Client"], CLIENT_ID)
            self.assertNotIn("User-Agent", client._config.default_headers)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
