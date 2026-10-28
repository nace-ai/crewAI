import json
import unittest
from types import SimpleNamespace

from nace_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneResponse, Usage

from crewai_nace import DecideError, DrexDecideTool, DrexGuardrail, drex_router


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


if __name__ == "__main__":
    unittest.main()
