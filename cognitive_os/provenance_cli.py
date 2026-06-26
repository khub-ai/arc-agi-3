"""cos-provenance — print the credence trail for a single hypothesis.

Installed as the ``cos-provenance`` console script. Reads an NDJSON
telemetry file, filters to the events whose subject (or payload
``hypothesis_id``) matches the supplied id, and prints them in
emission order.

Usage::

    cos-provenance run.ndjson h47
    cos-provenance run.ndjson h47 --since-step 1200
    cos-provenance run.ndjson h47 --json     # raw JSON entries

The ``--json`` mode writes one entry per line as a JSON object,
preserving the full payload — useful for piping into ``jq`` or feeding
a downstream analyser. The default mode is the compact one-line
human-readable rendering from :func:`provenance.format_entry`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .provenance import (
    ProvenanceEntry,
    format_entry,
    trail_from_ndjson,
)


def _filter_since(trail: List[ProvenanceEntry],
                  since_step: Optional[int]) -> List[ProvenanceEntry]:
    if since_step is None:
        return trail
    return [e for e in trail
            if e.step is not None and e.step >= since_step]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cos-provenance",
        description="Print the credence-update trail for one hypothesis.",
    )
    parser.add_argument("path", type=Path,
                        help="Path to the telemetry NDJSON file.")
    parser.add_argument("hypothesis_id", type=str,
                        help="Hypothesis id (e.g. 'h47').")
    parser.add_argument("--since-step", type=int, default=None,
                        help="Only include entries at or after this step.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit one JSON object per entry (full payload).")
    parser.add_argument("--summary", action="store_true",
                        help="After the trail, print a short totals line.")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"cos-provenance: file not found: {args.path}", file=sys.stderr)
        return 1

    trail = trail_from_ndjson(str(args.path), args.hypothesis_id)
    trail = _filter_since(trail, args.since_step)

    if not trail:
        print(f"(no events for {args.hypothesis_id} in {args.path})",
              file=sys.stderr)
        return 0

    if args.as_json:
        for entry in trail:
            sys.stdout.write(json.dumps(asdict(entry), separators=(",", ":")))
            sys.stdout.write("\n")
    else:
        for entry in trail:
            sys.stdout.write(format_entry(entry))
            sys.stdout.write("\n")

    if args.summary:
        n_support     = sum(1 for e in trail if e.direction == "support")
        n_contradict  = sum(1 for e in trail if e.direction == "contradict")
        n_decay       = sum(1 for e in trail if e.direction == "decay")
        n_specialise  = sum(1 for e in trail if e.direction == "specialise")
        last_after    = next((e.point_after for e in reversed(trail)
                              if e.point_after is not None), None)
        sys.stdout.write(
            f"# {len(trail)} entries  "
            f"support={n_support} contradict={n_contradict} "
            f"decay={n_decay} specialise={n_specialise}"
            + (f"  current={last_after:.3f}" if last_after is not None else "")
            + "\n"
        )
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
