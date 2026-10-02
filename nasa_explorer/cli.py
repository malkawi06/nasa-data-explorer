"""Command line: nasa-explore PATH [--var NAME] [--bbox W,S,E,N] [--start D --end D] [--ai] [--ask Q]"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

from .core import ReadOptions


def _bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox needs W,S,E,N e.g. -10,30,40,60")
    w, s, e, n = parts
    if s > n:
        raise argparse.ArgumentTypeError("--bbox south must be <= north")
    return w, s, e, n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nasa-explore", description="Detect, read and analyse almost any scientific data file."
    )
    p.add_argument("path", nargs="?", help="file, archive or folder")
    p.add_argument("--var", help="variable / column to focus on")
    p.add_argument("--bbox", type=_bbox, help="spatial subset W,S,E,N in degrees")
    p.add_argument("--start", help="start date, e.g. 2020-01-01")
    p.add_argument("--end", help="end date")
    p.add_argument("--out", default="reports", help="output folder (default: reports)")
    p.add_argument("--lang", choices=("en", "ar"), default="en", help="report / AI language")
    p.add_argument(
        "--ai", action="store_true", help="add AI summaries (needs Ollama or an API key)"
    )
    p.add_argument(
        "--ai-images",
        action="store_true",
        help="with --ai: also send a downscaled copy of plain images to the provider "
        "(Gemini/Anthropic) for a visual analysis",
    )
    p.add_argument(
        "--power",
        metavar="LAT,LON[,START,END]",
        help="download NASA POWER daily data for a point (default: last 10 full years) and analyse it",
    )
    p.add_argument("--ask", metavar="QUESTION", help="Q&A over everything processed in --out")
    p.add_argument("--no-plots", action="store_true", help="skip plots (faster)")
    p.add_argument("--formats", action="store_true", help="list supported formats and exit")
    p.add_argument("--app", action="store_true", help="launch the Streamlit drag-and-drop app")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def print_formats() -> None:
    from .registry import readers

    for r in sorted(readers(), key=lambda r: (r.category, r.name)):
        missing = r.missing()
        state = "ok" if not missing else f"disabled (pip install 'nasa-data-explorer[{r.extra}]')"
        print(f"{r.category:24} {r.name:15} {' '.join(r.extensions):45} {state}")


def _fetch_power(spec: str, out_dir: Path) -> Path:
    from . import power

    parts = [p.strip() for p in spec.split(",")]
    if len(parts) not in (2, 4):
        raise ValueError("--power needs LAT,LON or LAT,LON,START,END")
    start, end = (parts[2], parts[3]) if len(parts) == 4 else power.default_period()
    return power.fetch_daily(float(parts[0]), float(parts[1]), start, end, out_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    if not args.verbose:
        warnings.filterwarnings("ignore")
    if args.formats:
        print_formats()
        return 0
    if args.app:
        app = Path(__file__).with_name("app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    if args.power:
        try:
            args.path = str(_fetch_power(args.power, Path(args.out) / "downloads"))
        except Exception as exc:
            print(f"error: NASA POWER request failed: {exc}", file=sys.stderr)
            return 1
    if not args.path and not args.ask:
        build_parser().print_usage()
        return 2

    from .pipeline import process

    opts = ReadOptions(
        var=args.var,
        bbox=args.bbox,
        start=args.start,
        end=args.end,
        lang=args.lang,
        plots=not args.no_plots,
    )
    if args.path:
        try:
            if args.ai_images:
                os.environ["AI_SEND_IMAGES"] = "1"
            reports = process(args.path, opts, args.out, ai=args.ai, reuse=bool(args.ask))
        except FileNotFoundError as exc:
            print(f"error: not found: {exc}", file=sys.stderr)
            return 1
        for r in reports:
            status = f"ERROR {r.error}" if r.error else f"{r.reader:15} {r.kind}"
            print(f"{status:30} {r.file}  ->  {r.html_path}")
        print(f"\nindex: {Path(args.out) / 'index.html'}  ({len(reports)} report(s))")
    if args.ask:
        from .qa import ask

        print("\n" + ask(args.ask, args.out, lang=args.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
