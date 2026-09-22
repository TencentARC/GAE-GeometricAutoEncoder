"""Validate the released configs: parse them and resolve every ``target``.

Catches two failure modes that are easy to ship by accident:
  1. YAML that does not parse, or whose ${...} interpolations do not resolve.
  2. A ``target: a.b.C`` that no longer exists after the class renames.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omegaconf import OmegaConf  # noqa: E402

CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def check_target(dotted: str) -> tuple[bool, str]:
    module, _, cls = dotted.rpartition(".")
    try:
        mod = importlib.import_module(module)
    except Exception as e:
        return False, f"cannot import module {module}: {type(e).__name__}: {e}"
    if not hasattr(mod, cls):
        return False, f"module {module} has no attribute {cls}"
    return True, ""


def main() -> int:
    failures = 0
    for path in sorted(CFG_DIR.glob("*.yaml")):
        try:
            cfg = OmegaConf.load(path)
        except Exception as e:
            print(f"FAIL {path.name}: cannot parse: {e}")
            failures += 1
            continue

        # Force resolution of ${...} references.
        try:
            OmegaConf.resolve(cfg)   # in-place; raises on missing ${...}
            resolved = cfg
        except Exception as e:
            print(f"FAIL {path.name}: interpolation error: {type(e).__name__}: {e}")
            failures += 1
            continue

        targets = []
        for key in ("stage_1", "stage_2"):
            node = OmegaConf.select(resolved, key)
            if node is not None:
                t = OmegaConf.select(node, "target")
                if t:
                    targets.append((key, t))

        msgs = []
        for key, t in targets:
            ok, err = check_target(t)
            if not ok:
                msgs.append(f"{key}.target {t}: {err}")
                failures += 1

        latent = OmegaConf.select(resolved, "codec.latent_dim")
        in_ch = OmegaConf.select(resolved, "stage_2.params.in_channels")
        if latent is not None and in_ch is not None and latent != in_ch:
            msgs.append(f"latent_dim={latent} but in_channels={in_ch}")
            failures += 1

        if msgs:
            print(f"FAIL {path.name}:")
            for m in msgs:
                print(f"       - {m}")
        else:
            extra = f"latent_dim={latent}" if latent is not None else ""
            print(f"OK   {path.name}: {len(targets)} targets resolved  {extra}")

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
