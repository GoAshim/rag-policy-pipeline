"""
Automatic retrieval scoring: run every question in tests/test_questions.md
through the pipeline under several retrieval settings and score whether the
expected answer shows up in the results -- no hand-marking.

For each question we check the top-K chunks against the "Expected answer"
column: a chunk counts as a hit if it contains at least half of the expected
answer's 4-word phrases. That tolerates small formatting differences
(bullets, bold, line breaks) without letting unrelated chunks pass.

Short expected answers like "$1,000" are too short for phrase matching, so
they can be given an explicit pattern in MUST_PATTERNS below.

Usage (after build_index has run for the strategies you want):
    python -m scripts.score_retrieval
    python -m scripts.score_retrieval --strategies structure_aware --k 5
    python -m scripts.score_retrieval --extra tests/entity_questions.json

Prints a summary per strategy x setting and writes every question's rank to
tests/retrieval_scores.json so you can see exactly which questions moved.
"""

import argparse
import json
import re

from src import config, vectorstore
from src.pipeline import answer_question

TEST_QUESTIONS_PATH = config.PROJECT_ROOT / "tests" / "test_questions.md"
OUT_PATH = config.PROJECT_ROOT / "tests" / "retrieval_scores.json"

_CELL_BOUNDARY = re.compile(r"(?<!\\)\|")

# Question number -> regex(es) that must all appear in the same chunk.
MUST_PATTERNS = {
    "9": [r"Xifaxan[^\n]{0,300}?\$1,000\b"],
}

SETTINGS = {
    # name: (mode, alpha, keyword_filter)
    "vector (current)": ("vector", None, False),
    "hybrid 0.75": ("hybrid", 0.75, False),
    "hybrid 0.5": ("hybrid", 0.5, False),
    "vector + keyword tags": ("vector", None, True),
    "hybrid 0.75 + keyword tags": ("hybrid", 0.75, True),
}


def load_questions() -> list[dict]:
    qs = []
    for line in TEST_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| *\d+ *\|", line):
            continue
        cells = [c.strip() for c in _CELL_BOUNDARY.split(line)]
        num = cells[1]
        item = {"id": f"Q{num}", "q": cells[2], "expected": cells[3].replace("\\|", "|")}
        if num in MUST_PATTERNS:
            item["must"] = MUST_PATTERNS[num]
        qs.append(item)
    return qs


def _words(s: str) -> list[str]:
    s = re.sub(r"<br>|\*\*|__|#+|\\\|", " ", s.lower())
    return re.findall(r"[a-z0-9$][a-z0-9$,.%]*[a-z0-9%]|[a-z0-9]", s)


def _ngrams(ws: list[str], n: int = 4) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def judge(item: dict, texts: list[str]) -> int | None:
    """Rank (1-based) of the first chunk that contains the answer, else None."""
    for rank, text in enumerate(texts, 1):
        if "must" in item:
            flat = re.sub(r"[ \t]+", " ", text.replace("<br>", " ").replace("\\|", "|"))
            if all(re.search(p, flat, re.I) for p in item["must"]):
                return rank
        else:
            exp = _ngrams(_words(item["expected"]))
            if exp and len(exp & _ngrams(_words(text))) / len(exp) >= 0.5:
                return rank
    return None


def summarize(ranks: list[int | None], k: int) -> str:
    n = len(ranks)
    top1 = sum(r == 1 for r in ranks)
    top3 = sum(r is not None and r <= 3 for r in ranks)
    topk = sum(r is not None for r in ranks)
    mrr = sum(1 / r for r in ranks if r) / n
    return f"#1: {top1}/{n}  top3: {top3}/{n}  top{k}: {topk}/{n}  MRR {mrr:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["naive", "recursive", "structure_aware"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--extra", help="JSON list of {id, q, must:[regex]} extra questions")
    args = ap.parse_args()

    sets = {"test_questions.md": load_questions()}
    if args.extra:
        sets[args.extra] = json.load(open(args.extra))

    results = {}
    try:
        for strategy in args.strategies:
            for name, (mode, alpha, kw) in SETTINGS.items():
                for set_name, items in sets.items():
                    for item in items:
                        res = answer_question(strategy, item["q"], top_k=args.k,
                                              mode=mode, alpha=alpha, keyword_filter=kw)
                        texts = [c["text"] for c in res["retrieved_chunks"]]
                        results.setdefault(strategy, {}).setdefault(name, {})[item["id"]] = {
                            "set": set_name,
                            "rank": judge(item, texts),
                            "top": [f'{c["metadata"]["file_path"].rsplit("/", 1)[-1]}#{c["metadata"]["chunk_index"]}'
                                    for c in res["retrieved_chunks"]],
                        }
    finally:
        vectorstore.close_client()

    OUT_PATH.write_text(json.dumps(results, indent=1))
    for strategy, by_setting in results.items():
        print(f"\n== {strategy} ==")
        for set_name in sets:
            print(f"  [{set_name}]")
            for name, per_q in by_setting.items():
                ranks = [v["rank"] for v in per_q.values() if v["set"] == set_name]
                missed = [q for q, v in per_q.items() if v["set"] == set_name and v["rank"] is None]
                print(f"    {name:28s} {summarize(ranks, args.k)}   missed: {', '.join(missed) or '-'}")
    print(f"\nPer-question ranks written to {OUT_PATH}")


if __name__ == "__main__":
    main()
