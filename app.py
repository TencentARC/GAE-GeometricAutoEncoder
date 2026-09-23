"""GAE Hugging Face Space: camera-controlled I2V and prompt-to-image demos."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import spaces  # Provided by Hugging Face Spaces; optional for local smoke tests.
except ImportError:  # pragma: no cover - only used outside Spaces
    class _LocalSpaces:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _LocalSpaces()

import gradio as gr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.demo.trajectory_utils import (
    load_reference_poses, render_trajectory_preview, trajectory_for_preview,
)

HF_REPO = os.environ.get("GAE_HF_REPO", "TencentARC/GAE-D64-1B")
CKPT_DIR = Path(os.environ.get("GAE_SPACE_CKPT_DIR", "/tmp/gae-space-ckpts"))
OUTPUT_ROOT = Path(os.environ.get("GAE_SPACE_OUTPUT_DIR", "/tmp/gae-space-results"))

TRAJECTORIES = [
    ("Example camera poses — repository scale (recommended)", "example"),
    ("Wander — gentle turn-dominant motion", "wander"),
    ("Orbit — arc around the scene", "orbit"),
    ("Spiral — rising corkscrew", "spiral"),
    ("Drive — forward camera move", "drive"),
]
VIEW_CHOICES = [17, 33, 81]
# Uploaded images do not have a sibling *_poses.npz.  Use a shipped, moving
# 81-frame path as the fallback so the default trajectory has the same metric
# scale as a repository example instead of the nearly-static synthetic wander.
DEFAULT_EXAMPLE_POSES = ROOT / "examples" / "scenes" / "forest_lake_trail_poses.npz"


def _scene_examples() -> list[list[str]]:
    rows = []
    for image in sorted((ROOT / "examples" / "scenes").glob("*.jpg")):
        prompt_file = image.with_suffix(".txt")
        if prompt_file.is_file():
            rows.append([str(image), prompt_file.read_text(encoding="utf-8").strip(), "example"])
    return rows


def _t2i_examples() -> list[list[str]]:
    path = ROOT / "examples" / "t2i_prompts.txt"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([line])
    return rows


I2V_EXAMPLES = _scene_examples()
T2I_EXAMPLES = _t2i_examples()


def _duration_i2v(_image=None, _prompt="", _trajectory="wander", views: int = 17, steps: int = 25, *_args, **_kwargs) -> int:
    # This is a reservation hint for ZeroGPU; dedicated GPU Spaces can run longer.
    return min(900, max(120, int(90 + int(views) * int(steps) * 0.35)))


def _duration_t2i(_prompt="", steps: int = 25, *_args, **_kwargs) -> int:
    return min(600, max(120, int(90 + int(steps) * 4)))


def _latest(root: Path, suffixes: tuple[str, ...], reject: tuple[str, ...] = ()) -> Path | None:
    files = [
        p for suffix in suffixes for p in root.rglob(f"*{suffix}")
        if p.is_file() and not any(token in p.name for token in reject)
    ]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _pose_reference_for_image(image: str | None) -> Path | None:
    """Resolve the concrete pose path used for preview and metric scaling."""
    if image:
        image_path = Path(image)
        candidates = [
            image_path.with_name(f"{image_path.stem}_poses.npz"),
            ROOT / "examples" / "scenes" / f"{image_path.stem}_poses.npz",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return DEFAULT_EXAMPLE_POSES if DEFAULT_EXAMPLE_POSES.is_file() else None


def preview_i2v_trajectory(image: str | None, trajectory: str, views: int) -> str | None:
    """Render the exact selected input pose path before video generation."""
    reference = _pose_reference_for_image(image)
    if reference is None:
        return None
    try:
        reference_poses = load_reference_poses(reference, int(views))
        poses = trajectory_for_preview(reference_poses, trajectory, int(views))
        # Keep previews in a dedicated directory under the repository working
        # tree. Gradio allows files below the current working directory without
        # exposing the rest of OUTPUT_ROOT (which may contain other results).
        out_dir = ROOT / ".gradio_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{trajectory}-{int(views)}-{uuid.uuid4().hex}.png"
        render_trajectory_preview(
            poses, out_path,
            f"{trajectory} camera poses · {reference.stem} · {len(poses)} views",
        )
        return str(out_path)
    except (OSError, ValueError, ImportError) as exc:
        print(f"[space] trajectory preview failed: {exc}", flush=True)
        return None


def _run(command: list[str], output_dir: Path, timeout: int) -> tuple[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or "")[-3000:]
        raise gr.Error(f"Generation timed out after {timeout}s.\n\n{tail}") from exc
    log = output_dir / "space_run.log"
    log.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        tail = (result.stdout or "")[-5000:]
        raise gr.Error(f"Generation failed (exit {result.returncode}).\n\n{tail}")
    return result.stdout or "", time.time() - started


@spaces.GPU(duration=_duration_i2v)
def generate_i2v(
    image: str | None,
    prompt: str,
    trajectory: str,
    views: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    pc_stride: int,
):
    if not image:
        raise gr.Error("Upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please provide a scene description.")
    views, steps, seed = int(views), int(steps), int(seed)
    run_dir = OUTPUT_ROOT / f"i2v-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "demo" / "generate.py"),
        "--image", str(image),
        "--prompt", prompt,
        "--hf-repo", HF_REPO,
        "--ckpt-dir", str(CKPT_DIR),
        "--output", str(run_dir),
        "--num-views", str(views),
        "--total-views", str(views),
        "--sample-steps", str(steps),
        "--cfg-scale", str(float(cfg_scale)),
        "--seed", str(seed),
        "--pc-stride", str(int(pc_stride)),
    ]
    # Bundled scene examples use their matching poses. Synthetic alternatives
    # use the same pose file only as a metric-scale reference; their actual
    # path is generated by the selected motion.
    pose_file = _pose_reference_for_image(image)
    uses_bundled_poses = trajectory == "example" and pose_file is not None
    uses_example_poses = False
    if not uses_bundled_poses and trajectory == "example" and DEFAULT_EXAMPLE_POSES.is_file():
        pose_file = DEFAULT_EXAMPLE_POSES
        uses_example_poses = True
    if not uses_bundled_poses and not uses_example_poses:
        command += ["--free-rollout", "--trajectory", str(trajectory)]
        if pose_file is not None:
            command += ["--trajectory-reference-poses", str(pose_file)]
    elif uses_bundled_poses or uses_example_poses:
        command += ["--poses", str(pose_file)]
    _, elapsed = _run(command, run_dir, timeout=max(1800, _duration_i2v(views, steps) * 2))
    video = _latest(run_dir, (".mp4",))
    video = _latest(run_dir, ("_pred.mp4",)) or video
    depth_video = _latest(run_dir, ("_depth.mp4",))
    path_preview = _latest(run_dir, ("_trajectory.png",))
    depth = _latest(run_dir, ("_depth.png",))
    # I2V must expose the geometry decoded from the predicted latent.  The
    # evaluator also writes regen_from_video_pointcloud.ply; never return that
    # auxiliary reconstruction as the primary prediction.
    pointcloud = _latest(run_dir, ("_pred_pointcloud.ply",))
    if video is None:
        raise gr.Error("Generation completed but no MP4 was produced.")
    if uses_bundled_poses:
        mode = "matching repository camera poses"
    elif uses_example_poses:
        mode = f"canonical example camera poses ({pose_file.stem})"
    else:
        mode = f"synthetic {trajectory} trajectory"
    status = (
        f"GAE-64 · {views} views · {steps} Euler steps · seed {seed} · {mode} · "
        f"{elapsed:.1f}s\n\n[Download the full run log](file={run_dir / 'space_run.log'})"
    )
    return (
        str(video),
        str(depth_video) if depth_video else None,
        str(path_preview) if path_preview else None,
        str(depth) if depth else None,
        str(pointcloud) if pointcloud else None,
        status,
    )


@spaces.GPU(duration=_duration_t2i)
def generate_t2i(prompt: str, steps: int, cfg_scale: float, seed: int, pc_stride: int):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt first.")
    run_dir = OUTPUT_ROOT / f"t2i-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "demo" / "generate_t2i.py"),
        "--hf-repo", HF_REPO,
        "--ckpt-dir", str(CKPT_DIR),
        "--prompts", prompt,
        "--num-images", "1",
        "--sample-steps", str(int(steps)),
        "--cfg-scale", str(float(cfg_scale)),
        "--seed", str(int(seed)),
        "--pc-stride", str(int(pc_stride)),
        "--output", str(run_dir),
        "--save-pointcloud",
    ]
    _, elapsed = _run(command, run_dir, timeout=max(1200, _duration_t2i(steps) * 2))
    image = _latest(run_dir, (".png",), reject=("_depth",))
    depth = _latest(run_dir, ("_depth.png",))
    pointcloud = _latest(run_dir, ("_pointcloud.ply",))
    if image is None:
        raise gr.Error("Generation completed but no PNG was produced.")
    status = f"GAE-64 T2I · {steps} Euler steps · seed {int(seed)} · {elapsed:.1f}s"
    return str(image), str(depth) if depth else None, str(pointcloud) if pointcloud else None, status


@spaces.GPU(duration=900)
def reconstruct_vae(image: str | None):
    """Run codec-only reconstruction and expose RGB/depth outputs in the app."""
    if not image:
        raise gr.Error("Upload an image first.")
    run_dir = OUTPUT_ROOT / f"vae-recon-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "demo" / "reconstruct_vae.py"),
        "--image", str(image),
        "--hf-repo", HF_REPO,
        "--cache-dir", str(CKPT_DIR),
        "--output", str(run_dir),
    ]
    _, elapsed = _run(command, run_dir, timeout=1800)
    stem_dir = run_dir / Path(image).stem
    rgb = stem_dir / "rgb_recon.png"
    depth = stem_dir / "depth_recon.png"
    if not rgb.is_file():
        raise gr.Error("VAE reconstruction completed but no RGB output was produced.")
    return str(rgb), str(depth) if depth.is_file() else None, (
        f"GAE-64 VAE reconstruction · {elapsed:.1f}s\n\n"
        f"[Download the full run log](file={run_dir / 'space_run.log'})"
    )


CSS = """
#gae-container { max-width: 1180px; margin: 0 auto; }
"""

with gr.Blocks(title="GAE — Geometry-Native World Generation") as demo:
    with gr.Column(elem_id="gae-container"):
        gr.Markdown(
            """
# GAE — geometry-native world generation

**GAE** turns a still image and a prompt into a camera-controlled multi-view video,
with geometry decoded from the same latent state. The examples below are taken
from this repository's `examples/` directory and use the released
[`TencentARC/GAE-D64-1B`](https://huggingface.co/TencentARC/GAE-D64-1B) weights.
            """
        )
        with gr.Tabs():
            with gr.Tab("Image → camera-controlled video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(label="Input image", type="filepath", sources=["upload", "clipboard"], height=300)
                        i2v_prompt = gr.Textbox(label="Scene description", lines=4, placeholder="Describe the scene…")
                        i2v_trajectory = gr.Dropdown(label="Camera trajectory for uploaded images", choices=TRAJECTORIES, value="example")
                        i2v_pose_preview = gr.Image(
                            label="Selected input camera poses (preview)", type="filepath",
                            interactive=False, height=260,
                        )
                        i2v_run = gr.Button("Generate video", variant="primary")
                    with gr.Column(scale=1):
                        i2v_video = gr.Video(label="Generated RGB video", autoplay=True, loop=True, height=300)
                        i2v_depth_video = gr.Video(label="Decoded depth video", autoplay=True, loop=True, height=300)
                with gr.Row():
                    i2v_path = gr.Image(label="Decoded camera path", height=220)
                with gr.Row():
                    i2v_depth = gr.Image(label="Last decoded depth", height=220)
                    i2v_cloud = gr.File(label="Point cloud (.ply)")
                i2v_status = gr.Markdown()
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        i2v_views = gr.Dropdown(label="Views", choices=VIEW_CHOICES, value=17, type="value")
                        i2v_steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=5, value=25)
                        i2v_cfg = gr.Slider(label="CFG scale", minimum=1.0, maximum=4.0, step=0.1, value=2.0)
                    with gr.Row():
                        i2v_seed = gr.Number(label="Seed", value=42, precision=0)
                        i2v_stride = gr.Slider(label="Point-cloud stride", minimum=2, maximum=8, step=1, value=4)
                if I2V_EXAMPLES:
                    gr.Examples(
                        examples=I2V_EXAMPLES,
                        inputs=[i2v_image, i2v_prompt, i2v_trajectory],
                        label="Repository I2V examples — examples/scenes",
                        examples_per_page=8,
                    )
                preview_inputs = [i2v_image, i2v_trajectory, i2v_views]
                i2v_image.change(preview_i2v_trajectory, inputs=preview_inputs, outputs=i2v_pose_preview)
                i2v_image.input(preview_i2v_trajectory, inputs=preview_inputs, outputs=i2v_pose_preview)
                i2v_trajectory.change(preview_i2v_trajectory, inputs=preview_inputs, outputs=i2v_pose_preview)
                i2v_views.change(preview_i2v_trajectory, inputs=preview_inputs, outputs=i2v_pose_preview)
                i2v_run.click(
                    generate_i2v,
                    inputs=[i2v_image, i2v_prompt, i2v_trajectory, i2v_views, i2v_steps, i2v_cfg, i2v_seed, i2v_stride],
                    outputs=[i2v_video, i2v_depth_video, i2v_path, i2v_depth, i2v_cloud, i2v_status],
                )
            with gr.Tab("Text → image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe an image…")
                        t2i_run = gr.Button("Generate image", variant="primary")
                    with gr.Column(scale=1):
                        t2i_image = gr.Image(label="Generated image", height=320)
                        t2i_depth = gr.Image(label="Decoded depth", height=220)
                with gr.Row():
                    t2i_cloud = gr.File(label="Point cloud (.ply)")
                    t2i_status = gr.Markdown()
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        t2i_steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=5, value=25)
                        t2i_cfg = gr.Slider(label="CFG scale", minimum=1.0, maximum=4.0, step=0.1, value=2.0)
                        t2i_seed = gr.Number(label="Seed", value=0, precision=0)
                        t2i_stride = gr.Slider(label="Point-cloud stride", minimum=2, maximum=8, step=1, value=4)
                if T2I_EXAMPLES:
                    gr.Examples(
                        examples=T2I_EXAMPLES,
                        inputs=[t2i_prompt],
                        label="Repository T2I examples — examples/t2i_prompts.txt",
                        examples_per_page=8,
                    )
                t2i_run.click(
                    generate_t2i,
                    inputs=[t2i_prompt, t2i_steps, t2i_cfg, t2i_seed, t2i_stride],
                    outputs=[t2i_image, t2i_depth, t2i_cloud, t2i_status],
                )
            with gr.Tab("VAE reconstruction"):
                gr.Markdown(
                    "Encode an image with the GAE codec and decode it back to RGB "
                    "and depth. This tab does not run Flow/DiT generation."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vae_image = gr.Image(
                            label="Input image", type="filepath",
                            sources=["upload", "clipboard"], height=300,
                        )
                        vae_run = gr.Button("Reconstruct with VAE", variant="primary")
                    with gr.Column(scale=1):
                        vae_rgb = gr.Image(label="Reconstructed RGB", height=300)
                        vae_depth = gr.Image(label="Reconstructed depth", height=300)
                vae_status = gr.Markdown()
                vae_run.click(
                    reconstruct_vae,
                    inputs=[vae_image],
                    outputs=[vae_rgb, vae_depth, vae_status],
                )
        gr.Markdown(
            """
### Notes

- The first request downloads the GAE-64 codec, flow, latent statistics, and DA3 feature statistics from Hugging Face.
- Repository scene examples use their matching `*_poses.npz` camera paths; uploaded images use the selected synthetic trajectory.
- The free Space/ZeroGPU tier may time out on the full 81-view setting. Start with 17 views and 25 steps, then increase them on a GPU-backed Space.
            """
        )


if __name__ == "__main__":
    demo.launch()
