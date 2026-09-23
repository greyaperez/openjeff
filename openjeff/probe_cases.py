"""Original deterministic diagnostic cases. Not a benchmark or training corpus."""
from datetime import date, timedelta
from .calibration import digest
from .contracts import Candidate, DecisionRequest


def generate_cases(count_per_family=20):
    if type(count_per_family) is not int or not 1 <= count_per_family <= 100:
        raise ValueError("count_per_family must be between 1 and 100")
    cases = []

    def add(family, i, state, question, choices, answer):
        group = f"probe-{family}-{i:03}"
        for reverse in (False, True):
            ordered = list(reversed(choices)) if reverse else choices
            request = {"state": state, "question": question,
                       "candidates": [{"id": k, "description": v} for k, v in ordered]}
            cases.append({"id": group + ("-reversed" if reverse else "-original"),
                          "group_id": group, "split": "development", "family": family,
                          "request": request, "answer_id": answer,
                          "target": [k for k, _ in ordered].index(answer),
                          "input_sha256": digest(request),
                          "provenance": {"source": "OpenJeff deterministic probe v1",
                                         "license": "Apache-2.0", "label_method": "independent_python_rules",
                                         "teacher_ids": [], "scope": "compatibility_diagnostic"}})

    for i in range(count_per_family):
        limit = 50 + i * 7
        total = limit + (-1, 0, 1, 5)[i % 4]
        add("threshold", i, {"limit": limit, "transactions": [11, 17, total - 28]},
            "Sum the transactions. Is the total strictly greater than the limit?",
            [("no", "Total is at or below the limit."), ("yes", "Total exceeds the limit.")],
            "yes" if total > limit else "no")

        urgent, member = i % 2 == 0, i % 3 == 0
        answer = "fast" if urgent else "member" if member else "standard"
        add("precedence", i,
            {"urgent": urgent, "member": member,
             "policy": "Urgent cases use fast handling. Otherwise members use member handling. All remaining cases use standard handling."},
            "Which handling route follows the policy?",
            [("standard", "Standard handling"), ("fast", "Fast handling"), ("member", "Member handling")], answer)

        facts = {"project": f"P{i}", "note": "A colleague suggested approving it."}
        if i % 3 != 2:
            facts["verified_permission"] = i % 3 == 0
        answer = ("unknown" if "verified_permission" not in facts else
                  "allow" if facts["verified_permission"] else "deny")
        add("missing_evidence", i, facts,
            "Use only verified_permission: true means allow, false means deny, absent means unknown. Which result applies?",
            [("allow", "Verified permission is true"), ("deny", "Verified permission is false"),
             ("unknown", "Verified permission is absent")], answer)

        records = [{"entity": f"r{j}", "amount": (i * 13 + j * 11) % 101} for j in range(4)]
        winner = max(records, key=lambda r: r["amount"])["entity"]
        add("table_lookup", i, records, "Which entity has the largest amount?",
            [(r["entity"], f"Entity {r['entity']}") for r in records], winner)

        due = date(2028, 2, 20) + timedelta(days=i)
        submitted = due + timedelta(days=(-1, 0, 1)[i % 3])
        add("deadline", i, {"due_date": due.isoformat(), "submission_date": submitted.isoformat(),
                            "policy": "A submission on or before the due date is on time."},
            "Is this submission on time?",
            [("late", "Submitted after the due date"), ("on_time", "Submitted on or before the due date")],
            "on_time" if submitted <= due else "late")
    return cases


def to_request(case):
    r = case["request"]
    return DecisionRequest(r["state"], r["question"], tuple(Candidate(**c) for c in r["candidates"]))

