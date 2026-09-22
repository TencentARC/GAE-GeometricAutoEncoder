"""prepare_data.py must produce the training loader's packed schema."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

cv2 = pytest.importorskip("cv2")


def _write_re10k_scene(source: Path) -> None:
    scene = source / "process" / "train" / "0" / "scene1"
    scene.mkdir(parents=True)
    frames = []
    # Non-lexicographic stems verify integer-aware ordering.
    for stem, value in [("10", 30), ("2", 20), ("1", 10)]:
        cv2.imwrite(str(scene / f"{stem}.png"), np.full((28, 42, 3), value, np.uint8))
        frames.append({"file_path": stem,
                       "transform_matrix": np.eye(4, dtype=float)[:3].tolist()})
    (scene / "transforms.json").write_text(json.dumps(
        {"w": 42, "h": 28, "fl_x": 35, "fl_y": 35, "cx": 21, "cy": 14, "frames": frames}))
    (source / "train_caption.json").write_text(json.dumps({"scene1": "a test scene"}))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_re10k_schema(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _write_re10k_scene(source)
    output = tmp_path / "out"

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "prepare_data.py"), "re10k",
         "--source", str(source), "--output", str(output),
         "--split", "train", "--workers", "1", "--limit", "1"],
        check=True, cwd=ROOT,
    )

    meta = json.loads((output / "train" / "scene1" / "meta.json").read_text())
    assert (output / "train" / "scene1" / "video.mp4").is_file()
    assert isinstance(meta["rgb_resolution"], int)
    assert meta["rgb_resolution"] == 28
    assert meta["caption"] == "a test scene"
    assert [f["name"] for f in meta["frames"]] == ["1", "2", "10"]
    for frame in meta["frames"]:
        assert len(frame["c2w"]) == 4 and len(frame["c2w"][0]) == 4
