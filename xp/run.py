from __future__ import annotations

import argparse

from xp.protocol.phase import load_config

PHASES = {"bandit": "xp.bandit.phase"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run an XP phase (tune, freeze, sweep-evaluate).")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg["phase"] != args.phase:
        raise SystemExit(f"config phase {cfg['phase']!r} does not match --phase {args.phase!r}")
    module = __import__(PHASES[args.phase], fromlist=["run_phase"])
    module.run_phase(cfg, args.out)


if __name__ == "__main__":
    main()
