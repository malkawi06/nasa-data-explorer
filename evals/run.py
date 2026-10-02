"""Run the AI eval set against a real provider.

    python -m evals.run --provider gemini [--model gemini-3.8-flash] [--lang en]

Prints a table and writes evals/results/<provider>_<model>_<timestamp>.json.
Needs the provider's key in the environment / .env (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from nasa_explorer import ai
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file
from nasa_explorer.registry import read_file

from .cases import build_cases
from .score import score


def run(
    provider: str | None, model: str | None, lang: str, only: list[str] | None = None
) -> list[dict]:
    if model:
        os.environ["AI_MODEL"] = model
    client = ai.AIClient(ai.get_provider(provider), verbose=False)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in build_cases(Path(tmp)):
            if only and case.name not in only:
                continue
            opts = ReadOptions(lang=lang, plots=False)
            rep = process_file(case.path, opts, Path(tmp) / "reports")
            res, _, _ = read_file(case.path, opts)
            start, sent = time.time(), client.tokens_sent
            result = ai.run_workflow(
                ai.workflow_for(rep, res, lang, client.provider.name in ai.VISION), client
            )
            row = score(result, case)
            row.update(seconds=round(time.time() - start, 1), tokens=client.tokens_sent - sent)
            rows.append(row)
            print(
                f"{'PASS' if row['passed'] else 'FAIL'}  {case.name:22} precision={row['precision']:.2f} "
                f"facts={row['facts_hit']:.2f} terms={row['terms_hit']:.2f} rounds={row['rounds']} "
                f"wrong={row['mismatch']} unverified={row['unsupported']} {row['forbidden'] or ''}"
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--provider", help="ollama | gemini | groq | anthropic (default: first configured)"
    )
    p.add_argument("--model")
    p.add_argument("--lang", default="en", choices=("en", "ar"))
    p.add_argument("--case", action="append", help="run only this case (repeatable)")
    args = p.parse_args()
    rows = run(args.provider, args.model, args.lang, args.case)
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{len(rows)} cases passed")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    name = f"{args.provider or 'auto'}_{(args.model or 'default').replace('/', '-')}_{datetime.now():%Y%m%d-%H%M}.json"
    (out / name).write_text(
        json.dumps(
            {"provider": args.provider, "model": args.model, "lang": args.lang, "rows": rows},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"saved evals/results/{name}")


if __name__ == "__main__":
    main()
