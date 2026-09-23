import unittest
from collections import Counter
from openjeff.probe_cases import generate_cases, to_request
from openjeff.prompting import compile_request


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages) + "<answer>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class ProbeTests(unittest.TestCase):
    def test_paired_cases_preserve_answer_and_cover_families(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 200)
        self.assertEqual(len({r["id"] for r in cases}), 200)
        self.assertEqual(set(Counter(r["group_id"] for r in cases).values()), {2})
        self.assertEqual(len({r["family"] for r in cases}), 5)
        for first, second in zip(cases[::2], cases[1::2]):
            self.assertEqual(first["answer_id"], second["answer_id"])
            for row in (first, second):
                request = to_request(row)
                self.assertEqual(request.candidates[row["target"]].id, row["answer_id"])
                compiled = compile_request(CharTokenizer(), request)
                self.assertEqual(len(compiled["code_ids"]), len(request.candidates))

    def test_threshold_and_calendar_labels_have_edge_cases(self):
        rows = [r for r in generate_cases() if r["id"].endswith("original")]
        for row in rows:
            state = row["request"]["state"]
            if row["family"] == "threshold":
                self.assertEqual(row["answer_id"] == "yes", sum(state["transactions"]) > state["limit"])
            if row["family"] == "deadline":
                self.assertEqual(row["answer_id"] == "on_time", state["submission_date"] <= state["due_date"])

    def test_no_truncation(self):
        with self.assertRaises(ValueError):
            compile_request(CharTokenizer(), to_request(generate_cases(1)[0]), max_tokens=2)

    def test_multi_token_code_fails(self):
        class BadTokenizer(CharTokenizer):
            def encode(self, text, **kwargs):
                return super().encode(text, **kwargs) + ([0] if text[-1] in 'ABCD' else [])
        with self.assertRaises(ValueError):
            compile_request(BadTokenizer(), to_request(generate_cases(1)[0]))


if __name__ == "__main__":
    unittest.main()
