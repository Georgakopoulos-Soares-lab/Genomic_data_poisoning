#!/usr/bin/env python3
"""
Read an experiment YAML config and emit flattened KEY=VALUE lines for bash.

Usage in a bash script:
    eval "$(python scripts/parse_config.py configs/experiments/poison_12bp.yaml)"
    echo "$TRIGGER_SEQUENCE"   # → ACGTACGTACGT

Nested keys are flattened with underscores and uppercased:
    trigger.sequence  →  TRIGGER_SEQUENCE
    training.lr       →  TRAINING_LR
    paths.extracted   →  PATHS_EXTRACTED
"""

import shlex
import sys
import yaml


def flatten(d, prefix=""):
    """Recursively flatten a dict into PREFIX_KEY=value pairs.

    Lists of dicts get indexed keys:  triggers[0].sequence → TRIGGERS_0_SEQUENCE
    Plus a count key:                 triggers → TRIGGERS_COUNT=2
    Plain lists become comma-separated: tags: [a,b] → TAGS='a,b'
    """
    items = {}
    for k, v in d.items():
        key = f"{prefix}_{k}".upper() if prefix else k.upper()
        if isinstance(v, dict):
            items.update(flatten(v, key))
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                # List of dicts → indexed keys + count
                items[f"{key}_COUNT"] = len(v)
                for i, elem in enumerate(v):
                    items.update(flatten(elem, f"{key}_{i}"))
            else:
                # Plain list → comma-separated
                items[key] = ",".join(str(x) for x in v)
        else:
            items[key] = v
    return items


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    for key, val in sorted(flatten(cfg).items()):
        # Convert Python types to bash-friendly strings
        if val is None:
            val_str = ""
        elif isinstance(val, bool):
            val_str = "true" if val else "false"
        else:
            val_str = str(val)
        # Shell-escape the value
        print(f"{key}={shlex.quote(val_str)}")


if __name__ == "__main__":
    main()
