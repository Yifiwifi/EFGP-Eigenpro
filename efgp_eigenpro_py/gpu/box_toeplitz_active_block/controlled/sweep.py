from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import ControlledConfig, _sanitize_json, run_controlled_experiment


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    if not str(raw).strip():
        return ()
    try:
        return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floating-point values") from exc


def _load_config(path: str) -> ControlledConfig:
    if not path:
        return ControlledConfig()
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ControlledConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown ControlledConfig fields in {path}: {unknown}")
    if "methods" in payload:
        payload["methods"] = tuple(payload["methods"])
    if "diagnostic_topk" in payload:
        payload["diagnostic_topk"] = tuple(payload["diagnostic_topk"])
    return ControlledConfig(**payload)


def _tag(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def run_one_factor_sweep(
    base: ControlledConfig,
    *,
    lambdas: tuple[float, ...],
    lengthscales: tuple[float, ...],
    output_root: Path,
) -> Path:
    """Run one-factor-at-a-time systems; never form a Cartesian product."""
    output_root.mkdir(parents=True, exist_ok=True)
    settings: list[tuple[str, float, ControlledConfig]] = [
        (
            "reference",
            1.0,
            replace(base, output_dir=str(output_root / "reference")),
        )
    ]
    for value in lambdas:
        if float(value) == float(base.reg_lambda):
            continue
        settings.append(
            (
                "reg_lambda",
                float(value),
                replace(
                    base,
                    reg_lambda=float(value),
                    output_dir=str(output_root / f"lambda_{_tag(value)}"),
                ),
            )
        )
    for value in lengthscales:
        if float(value) == float(base.lengthscale):
            continue
        settings.append(
            (
                "lengthscale",
                float(value),
                replace(
                    base,
                    lengthscale=float(value),
                    output_dir=str(output_root / f"lengthscale_{_tag(value)}"),
                ),
            )
        )

    index_rows: list[dict[str, Any]] = []
    for factor, value, cfg in settings:
        run_dir = run_controlled_experiment(cfg)
        summary = json.loads((run_dir / "matched_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "system_manifest.json").read_text(encoding="utf-8"))
        for row in summary:
            index_rows.append(
                {
                    "factor": factor,
                    "factor_value": value,
                    "reg_lambda": float(cfg.reg_lambda),
                    "lengthscale": float(cfg.lengthscale),
                    "M": manifest.get("M"),
                    "system_id": manifest.get("system_id"),
                    "run_dir": str(run_dir),
                    **row,
                }
            )

    (output_root / "sweep_config.json").write_text(
        json.dumps(_sanitize_json(asdict(base)), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (output_root / "sweep_index.json").write_text(
        json.dumps(_sanitize_json(index_rows), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    columns = sorted({key for row in index_rows for key in row})
    with (output_root / "sweep_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in index_rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                    for key, value in row.items()
                }
            )
    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-factor-at-a-time lambda/lengthscale sweep using the matched-system runner."
        )
    )
    parser.add_argument(
        "--config",
        default="",
        help="JSON object with ControlledConfig fields; defaults to the demo configuration.",
    )
    parser.add_argument("--lambdas", type=_parse_float_tuple, default=(0.01, 0.1, 1.0))
    parser.add_argument(
        "--lengthscales", type=_parse_float_tuple, default=(0.05, 0.1, 0.2)
    )
    parser.add_argument("--output-root", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base = _load_config(args.config)
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        tag = datetime.now().strftime("sensitivity_%Y%m%d_%H%M%S")
        output_root = (Path(__file__).resolve().parent / "outputs" / tag).resolve()
    result = run_one_factor_sweep(
        base,
        lambdas=tuple(args.lambdas),
        lengthscales=tuple(args.lengthscales),
        output_root=output_root,
    )
    print(f"Wrote one-factor sweep index to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
