"""Interactive camera authoring UI, launched by the resident inference service."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
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
from scripts.demo import captioner
from scripts.demo.trajectory_utils import (
    load_reference_poses,
    render_trajectory_preview,
    trajectory_for_preview,
)

RESIDENT_ENGINE = None  # Set only after service initialization and warmup.
_SCENE_CACHE: OrderedDict[str, dict] = OrderedDict()
_SCENE_CACHE_LOCK = threading.Lock()
_LATEST_SCENE_REQUEST: dict[str, str] = {}

# Measure the actual browser interval, including queueing, transfer and decode.
PLAYBACK_TIMING_JS = r"""(...args) => {
    const metric = document.getElementById('gae-client-latency');
    const container = document.getElementById('gae-paired-result');
    const bar = document.getElementById('gae-request-progress');
    if (!metric || !container) return args;
    const started = performance.now();
    const token = String(started);
    let cleared = !container.querySelector('video');
    metric.dataset.request = token;
    window.gaeCameraEditor?.setGenerating?.(true);
    const views = Number(args[3]) || 81, steps = Number(args[4]) || 25;
    const config = `${views}:${steps}`;
    const measured = metric.dataset.lastConfig === config ? Number(metric.dataset.lastSeconds) : 0;
    const estimate = measured > 0 ? measured * 1.1 : Math.max(8, 6 + 34 * views / 81 * steps / 25);
    let finished = false;
    let timer, ticker;
    const cleanup = () => { observer.disconnect(); clearTimeout(timer); clearInterval(ticker); };
    const updateEstimate = () => {
        if (metric.dataset.request !== token) { cleanup(); return; }
        const elapsed = (performance.now() - started) / 1000;
        const remaining = Math.ceil(Math.max(0, estimate - elapsed));
        metric.textContent = remaining > 0
            ? `Estimated ready in about ${remaining} s · elapsed ${elapsed.toFixed(0)} s`
            : `Still generating · elapsed ${elapsed.toFixed(0)} s · initial estimate exceeded`;
        if (bar) { bar.hidden = false; bar.value = Math.min(95, elapsed / estimate * 100); }
    };
    const ready = (video) => {
        if (finished || metric.dataset.request !== token) { cleanup(); return; }
        if (video !== container.querySelector('video')) return;
        if (!cleared || !video.currentSrc || video.readyState < 2) return;
        finished = true;
        window.gaeCameraEditor?.setGenerating?.(false);
        cleanup();
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (metric.dataset.request === token) {
                const actual = (performance.now() - started) / 1000;
                metric.dataset.lastConfig = config;
                metric.dataset.lastSeconds = String(actual);
                metric.textContent = `Ready to play in ${actual.toFixed(2)} s · browser end-to-end`;
                if (bar) bar.value = 100;
            }
        }));
    };
    const inspect = () => {
        if (metric.dataset.request !== token) { cleanup(); return; }
        const video = container.querySelector('video');
        if (!video || !video.getAttribute('src')) cleared = true;
        if (video) {
            video.addEventListener('loadeddata', () => ready(video), {once: true});
            ready(video);
        }
    };
    const observer = new MutationObserver(inspect);
    observer.observe(container, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    ticker = setInterval(updateEstimate, 250);
    updateEstimate();
    timer = setTimeout(() => {
        cleanup();
        if (!finished && metric.dataset.request === token)
            metric.textContent = 'Still waiting for playback. Check the generation status below.';
    }, 600000);
    inspect();
    return args;
}"""

HF_REPO = os.environ.get("GAE_HF_REPO", "TencentARC/GAE-D64-1B")
# Same directory the resident engine loads from, so side jobs never re-download.
CKPT_DIR = Path(
    os.environ.get("GAE_SPACE_CKPT_DIR") or os.environ.get("GAE_CKPT_DIR") or str(ROOT / "ckpts")
)
OUTPUT_ROOT = Path(os.environ.get("GAE_SPACE_OUTPUT_DIR", "/tmp/gae-space-results"))

VIEW_CHOICES = [17, 33, 81]
# Uploaded images do not have a sibling *_poses.npz.  Use a shipped, moving
# 81-frame path as the fallback so the default trajectory has the same metric
# scale as a repository example instead of a nearly-static synthetic path.
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


def _vae_video_examples() -> list[list[str]]:
    path = ROOT / "examples" / "recon_videos"
    # Representative real clips from the d64 step-0018500 evaluation sweep,
    # renamed by scene content for a readable reconstruction gallery.
    names = (
        "historic_courtyard.mp4",
        "autumn_waterfall.mp4",
        "offroad_vehicle.mp4",
        "bathroom_hallway.mp4",
        "office_shelf.mp4",
        "airport_luggage_vehicle.mp4",
    )
    return [[str(path / name)] for name in names if (path / name).is_file()]


VAE_VIDEO_EXAMPLES = _vae_video_examples()


def _duration_i2v(_image=None, _prompt="", _trajectory="forward", views: int = 17, steps: int = 25, *_args, **_kwargs) -> int:
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


def _camera_editor_preset(image: str | None) -> dict | None:
    """Convert an example's dataset c2w path to editable Studio keyframes."""
    if not image:
        return None
    image_path = Path(image)
    candidates = [
        image_path.with_name(f"{image_path.stem}_poses.npz"),
        ROOT / "examples" / "scenes" / f"{image_path.stem}_poses.npz",
    ]
    pose_path = next((path for path in candidates if path.is_file()), None)
    if pose_path is None:
        return None
    try:
        import numpy as np

        poses = load_reference_poses(pose_path, 81)
        relative = np.linalg.inv(poses[0]) @ poses
        indices = np.linspace(0, len(relative) - 1, min(6, len(relative))).round().astype(int)
        keyframes = []
        for order, index in enumerate(indices):
            transform = relative[index]
            rotation = transform[:3, :3]
            yaw = float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))
            pitch = float(np.degrees(np.arctan2(-rotation[1, 2], rotation[1, 1])))
            position = transform[:3, 3]
            role = "start" if order == 0 else "end" if order == len(indices) - 1 else "keyframe"
            keyframes.append(dict(
                id="start" if order == 0 else f"preset-{order}",
                role=role,
                time=float(index / max(len(relative) - 1, 1)),
                x=0.0 if order == 0 else float(position[0]),
                y=0.0 if order == 0 else float(position[1]),
                z=0.0 if order == 0 else float(position[2]),
                yaw=0.0 if order == 0 else yaw,
                pitch=0.0 if order == 0 else pitch,
            ))
        return {"source": pose_path.name, "keyframes": keyframes}
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"[camera] could not load preset {pose_path}: {exc}", flush=True)
        return None


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


def _camera_preview_path(prefix):
    directory = ROOT / ".gradio_previews"
    directory.mkdir(exist_ok=True)
    return directory / f"{prefix}-{uuid.uuid4().hex}.png"


def _session_key(request: gr.Request) -> str:
    return request.session_hash or "anonymous"


def mark_scene_request(_image, request: gr.Request):
    """Invalidate any slower scene preparation already running for this tab."""
    token = uuid.uuid4().hex
    _LATEST_SCENE_REQUEST[_session_key(request)] = token
    return token


def prepare_direct_camera(image, token, request: gr.Request):
    if not image:
        return None, None
    if RESIDENT_ENGINE is None:
        raise gr.Error("The camera editor requires the resident GPU service.")
    from scripts.demo.direct_camera import image_key

    key = image_key(image)
    with _SCENE_CACHE_LOCK:
        scene = _SCENE_CACHE.get(key)
        if scene is not None:
            _SCENE_CACHE.move_to_end(key)
    if scene is None:
        scene = RESIDENT_ENGINE.prepare_camera_scene(image)
        scene["presetPath"] = _camera_editor_preset(image)
        with _SCENE_CACHE_LOCK:
            _SCENE_CACHE[key] = scene
            _SCENE_CACHE.move_to_end(key)
            while len(_SCENE_CACHE) > 3:
                _SCENE_CACHE.popitem(last=False)
    if _LATEST_SCENE_REQUEST.get(_session_key(request)) != token:
        return gr.skip(), gr.skip()
    state = {k: scene[k] for k in ("imageKey", "pivotDepth", "K", "preparedImage", "metricScale", "scaleQuantiles")}
    return scene, state


def generate_direct_camera(image, prompt, _trajectory, views, steps, cfg, seed, stride, payload, scene):
    from scripts.demo.direct_camera import image_key, end_view_poses
    import numpy as np
    if not image or not scene or image_key(image) != scene.get("imageKey"):
        raise gr.Error("Wait for this image to finish preparing its camera preview.")
    try:
        poses = end_view_poses(payload, scene, views)
    except (ValueError, TypeError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc
    path = _camera_preview_path("direct-view").with_suffix(".npz")
    K = np.asarray(scene["K"], np.float32)
    np.savez(path, c2w=poses, K=np.repeat(K[None], len(poses), axis=0), fps=12,
             image_preprocessed=True, metric_scale=float(scene.get("metricScale", 1)),
             source_pivot_depth=float(scene["pivotDepth"]),
             requested_camera=payload if isinstance(payload, str) else json.dumps(payload))
    # Existing generator handles the same fixed poses for resident and CLI routes.
    yield from generate_i2v(scene.get("preparedImage", image), (prompt or "").strip() or "A detailed view of the scene shown in the input image.", "example", views, steps, cfg, seed, stride,
                           camera_pose_file=str(path))

def _run(command: list[str], output_dir: Path, timeout: int) -> tuple[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Runs beside the resident engine: one GPU, the one with the most free memory.
    env = captioner.side_job_env(os.environ)
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


def _run_i2v_stream(command, output_dir, timeout):
    """Yield only explicitly completed RGB artifacts, while geometry continues."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", EVAL_STAGE_OUTPUT_LOCAL="0")
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"
    started = time.monotonic()
    messages = queue.Queue()
    with (output_dir / "space_run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        def read_output():
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.startswith("GAE_ARTIFACT_READY "):
                    try:
                        path = Path(json.loads(line.split(" ", 1)[1])["path"])
                        if path.name.endswith("_pred.mp4") and path.is_relative_to(output_dir):
                            messages.put(str(path))
                    except (ValueError, KeyError):
                        pass
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while proc.poll() is None or reader.is_alive() or not messages.empty():
                if time.monotonic()-started > timeout:
                    raise gr.Error(f"Generation timed out after {timeout}s.")
                try:
                    path = messages.get(timeout=.1)
                except queue.Empty:
                    continue
                yield {"video": path, "elapsed": time.monotonic()-started, "error": None}
            error = None if proc.returncode == 0 else f"Post-processing exited with code {proc.returncode}; see the run log."
            yield {"video": None, "elapsed": time.monotonic()-started, "error": error}
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            reader.join(timeout=5)
            proc.stdout.close()


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
    camera_pose_file: str | None = None,
):
    if not image:
        raise gr.Error("Upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please provide a scene description.")
    views, steps, seed = int(views), int(steps), int(seed)
    yield None, None, None, None, None, "Generating your scene…", None, None
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
        "--no-pointcloud", "--preview-geometry",
    ]
    if os.environ.get("GAE_PREVIEW_RGB_DEVICE"):
        command += ["--preview-rgb-device", os.environ["GAE_PREVIEW_RGB_DEVICE"]]
    # Bundled scene examples use their matching poses. Synthetic alternatives
    # use the same pose file only as a metric-scale reference; their actual
    # path is generated by the selected motion.
    pose_file = _pose_reference_for_image(image)
    controlled = camera_pose_file is not None
    if camera_pose_file is not None:
        pose_file = Path(camera_pose_file)
    uses_bundled_poses = trajectory == "example" and pose_file is not None
    uses_example_poses = False
    if not uses_bundled_poses and trajectory == "example" and DEFAULT_EXAMPLE_POSES.is_file():
        pose_file = DEFAULT_EXAMPLE_POSES
        uses_example_poses = True
    if controlled:
        command += ["--poses", str(pose_file)]
    elif not uses_bundled_poses and not uses_example_poses:
        command += ["--free-rollout", "--trajectory", str(trajectory)]
        if pose_file is not None:
            command += ["--trajectory-reference-poses", str(pose_file)]
    elif uses_bundled_poses or uses_example_poses:
        command += ["--poses", str(pose_file)]
    if RESIDENT_ENGINE is not None:
        result = RESIDENT_ENGINE.generate(
            image, prompt, poses=pose_file if (controlled or uses_bundled_poses or uses_example_poses) else None,
            trajectory=trajectory, reference_poses=pose_file,
            views=views, steps=steps, cfg_scale=float(cfg_scale), seed=seed,
            stride=int(pc_stride), output_dir=run_dir,
        )
        elapsed = result["timings"]["request_seconds"]
        status = (f"GAE-64 · {views} views · {steps} Euler steps · seed {seed} · "
                  f"Synchronized result in {elapsed:.1f}s (server)")
        yield (result["rgb_video"], result.get("depth_video"), None, None, None, status,
               result["synchronized_video"], result["geometry"])
        return
    elapsed, error = 0., None
    for event in _run_i2v_stream(command, run_dir, timeout=max(1800, _duration_i2v(views, steps) * 2)):
        elapsed, error = event["elapsed"], event["error"]
        if event["video"]:
            yield None, None, None, None, None, "Preparing synchronized RGB + 3D…", None, None
    video = _latest(run_dir, ("_pred.mp4",))
    depth_video = _latest(run_dir, ("_depth.mp4",))
    path_preview = _latest(run_dir, ("_trajectory.png",))
    depth = _latest(run_dir, ("_depth.png",))
    # I2V must expose the geometry decoded from the predicted latent.  The
    # evaluator also writes regen_from_video_pointcloud.ply; never return that
    # auxiliary reconstruction as the primary prediction.
    if video is None:
        raise gr.Error(error or "Generation completed but no RGB video was produced.")
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
    if error:
        status += f" · {error}"
    preview_source = _latest(run_dir, ("_preview.npz",))
    values = (
        str(video), str(depth_video) if depth_video else None,
        str(path_preview) if path_preview else None, str(depth) if depth else None,
        None, status, None, str(preview_source) if preview_source else None,
    )
    # Publish one paired result; independent autoplay players would drift.
    yield None, None, None, None, None, "Preparing synchronized RGB + 3D…", None, None
    if preview_source is None:
        yield (*values[:5], status + " · 3D preview unavailable; see the run log.", *values[6:])
        return
    try:
        from scripts.demo.progressive_preview import render_preview
        preview, timing = render_preview(
            preview_source, preview_source.with_name(preview_source.stem + "_synchronized.mp4"),
            rgb_video=video, device=os.environ.get("GAE_PREVIEW_DEVICE", "auto"))
    except Exception as exc:  # noqa: BLE001 - preserve completed RGB on preview failures
        (run_dir / "preview_error.log").write_text(str(exc), encoding="utf-8")
        yield (*values[:5], status + " · 3D preview failed; RGB is ready.", *values[6:])
        return
    total = elapsed + timing["total_seconds"]
    yield (*values[:5], status + f" · Synchronized preview {timing['total_seconds']:.1f}s · total {total:.1f}s",
           str(preview), str(preview_source))


def download_pointcloud(source):
    if not source:
        raise gr.Error("Generate a scene first.")
    source = Path(source).resolve()
    if not source.is_relative_to(OUTPUT_ROOT.resolve()) or source.suffix != ".npz":
        raise gr.Error("Invalid scene geometry.")
    from scripts.demo.progressive_preview import export_ply
    return str(export_ply(source))



@spaces.GPU(duration=_duration_t2i)
def generate_t2i(prompt: str, steps: int, guidance: str, scale: float, seed: int, pc_stride: int):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt first.")
    guidance = "cfg" if str(guidance).lower().startswith("cfg") else "ig"
    run_dir = OUTPUT_ROOT / f"t2i-{uuid.uuid4().hex}"
    label = f"{guidance.upper()} {float(scale):g}"
    if RESIDENT_ENGINE is not None:
        started = time.perf_counter()
        written = RESIDENT_ENGINE.text_to_image(
            prompt, steps=int(steps), guidance=guidance, scale=float(scale), seed=int(seed),
            pc_stride=int(pc_stride), output_dir=run_dir)
        status = (f"GAE-64 T2I · {steps} Euler steps · {label} · seed {int(seed)} · "
                  f"{time.perf_counter() - started:.1f}s on the resident GPUs")
        return (str(written["image"]), str(written["depth"]), str(written["pointcloud"]),
                str(written["preview"]), status)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "demo" / "generate_t2i.py"),
        "--hf-repo", HF_REPO,
        "--ckpt-dir", str(CKPT_DIR),
        "--prompts", prompt,
        "--num-images", "1",
        "--sample-steps", str(int(steps)),
        "--guidance", guidance,
        "--cfg-scale" if guidance == "cfg" else "--ig-scale", str(float(scale)),
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
    status = f"GAE-64 T2I · {steps} Euler steps · {label} · seed {int(seed)} · {elapsed:.1f}s"
    return (str(image), str(depth) if depth else None, str(pointcloud) if pointcloud else None,
            _cloud_view(pointcloud), status)


def _cloud_view(ply: Path | None) -> str | None:
    if ply is None:
        return None
    try:
        from scripts.demo.pointcloud_view import ply_preview
        preview = ply_preview(ply)
    except Exception as exc:  # noqa: BLE001 - the .ply download still works
        print(f"[pointcloud] preview failed for {ply}: {type(exc).__name__}: {exc}", flush=True)
        return None
    return str(preview) if preview else None


def _reconstruct_resident(video: Path, out_dir: Path) -> dict[str, Path] | None:
    """Reuse the resident codec; None when the service has none or runs out of memory."""
    engine = RESIDENT_ENGINE
    if engine is None:
        return None
    import torch
    from scripts.demo.reconstruct_vae import reconstruct
    with engine.lock:
        try:
            return reconstruct(engine.model, video, out_dir, pc_stride=4)
        except torch.cuda.OutOfMemoryError:
            print("[vae] resident GPU out of memory; using a separate process", flush=True)
            return None
        finally:
            torch.cuda.empty_cache()


@spaces.GPU(duration=900)
def reconstruct_vae(image: str | None):
    """Run codec-only reconstruction and expose RGB/depth/point-cloud outputs."""
    if not image:
        raise gr.Error("Upload an image first.")
    run_dir = OUTPUT_ROOT / f"vae-recon-{uuid.uuid4().hex}"
    stem_dir = run_dir / Path(image).stem
    started = time.perf_counter()
    written = _reconstruct_resident(Path(image), stem_dir)
    elapsed = time.perf_counter() - started
    log_note = ""
    if written is None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "demo" / "reconstruct_vae.py"),
            "--video", str(image),
            "--hf-repo", HF_REPO,
            "--cache-dir", str(CKPT_DIR),
            "--pc-stride", "4",
            "--output", str(run_dir),
        ]
        _, elapsed = _run(command, run_dir, timeout=1800)
        written = {kind: stem_dir / name for kind, name in (
            ("rgb", "rgb_recon.mp4"), ("depth", "depth_recon.mp4"),
            ("pointcloud", "recon_pointcloud.ply"), ("preview", "recon_pointcloud_preview.glb"))}
        log_note = f"\n\n[Download the full run log](file={run_dir / 'space_run.log'})"
    rgb, depth = written.get("rgb"), written.get("depth")
    pointcloud, preview = written.get("pointcloud"), written.get("preview")
    if rgb is None or not rgb.is_file():
        raise gr.Error("VAE reconstruction completed but no RGB output was produced.")
    if pointcloud is None or not pointcloud.is_file():
        raise gr.Error("VAE reconstruction completed but no point cloud was produced.")
    return (
        str(rgb),
        str(depth) if depth and depth.is_file() else None,
        str(pointcloud),
        str(preview) if preview and preview.is_file() else None,
        f"GAE-64 VAE reconstruction · {elapsed:.1f}s{log_note}",
    )


def describe_image(image: str | None):
    """Stream a scene description for the uploaded image into the prompt box."""
    if not image:
        yield gr.update()
        return
    try:
        yield from captioner.stream(image)
    except Exception as exc:  # noqa: BLE001 - the prompt stays editable
        print(f"[caption] failed: {type(exc).__name__}: {exc}", flush=True)
        gr.Warning("Automatic description failed; type one if you want to guide the scene.")
        yield gr.update()


def choose_scene_example(event: gr.SelectData, request: gr.Request):
    index = int(event.index)
    if not 0 <= index < len(I2V_EXAMPLES):
        raise gr.Error("Choose an example image.")
    token = mark_scene_request(I2V_EXAMPLES[index][0], request)
    return I2V_EXAMPLES[index][0], I2V_EXAMPLES[index][1], token


CSS = """
#gae-container { max-width: 1700px; margin: 0 auto; }
#gae-output-panel { position: sticky; top: 14px; align-self: flex-start; }
#gae-paired-result { min-height: 260px; }
@media (max-width: 1000px) { #gae-output-panel { position: static; } }
"""

with gr.Blocks(title="GAE Camera Studio") as demo:
    with gr.Column(elem_id="gae-container"):
        gr.Markdown(
            """
# GAE · Camera Studio

Choose an image, place your cameras, and generate a scene in motion.
            """
        )
        with gr.Tabs(selected="i2v"):
            with gr.Tab("Camera Studio", id="i2v"):
                with gr.Row(equal_height=False, elem_id="gae-io-layout"):
                    with gr.Column(scale=3, min_width=550, elem_id="gae-input-panel"):
                        i2v_image = gr.Image(label="Input image", type="filepath", sources=["upload", "clipboard"], height=220)
                        with gr.Accordion("Scene description · filled in automatically", open=True):
                            i2v_prompt = gr.Textbox(label="Scene description", lines=3,
                                                    placeholder="Written automatically after upload; edit freely…")
                        gr.Markdown("**Build your camera path** · Move, add keyframes, then choose Final.")
                        i2v_camera_editor = gr.HTML(
                            value=None,
                            html_template='<div></div>',
                            css_template="",
                            js_on_load=(ROOT / "scripts/demo/camera_editor_loader.js").read_text().replace("__ASSET_ROOT__", str(ROOT / "scripts/demo")),
                            elem_id="gae-camera-editor")
                        i2v_camera_scene = gr.State()
                        i2v_scene_token = gr.State("")
                        i2v_camera_payload = gr.Textbox(visible=False, value="")
                        i2v_trajectory = gr.State("direct-view")
                        i2v_load_preset = gr.Button(
                            "Use example default trajectory",
                            elem_id="gae-load-preset",
                        )
                        i2v_run = gr.Button("Generate video", variant="primary", elem_id="gae-generate")
                        with gr.Accordion("Advanced settings", open=False):
                            with gr.Row():
                                i2v_views = gr.Dropdown(label="Views", choices=VIEW_CHOICES, value=81, type="value")
                                i2v_steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=5, value=25)
                                i2v_cfg = gr.Slider(label="CFG scale", minimum=1.0, maximum=4.0, step=0.1, value=2.0)
                            with gr.Row():
                                i2v_seed = gr.Number(label="Seed", value=42, precision=0)
                                i2v_stride = gr.Slider(label="3D preview sampling stride", minimum=1, maximum=8, step=1, value=2)
                        if I2V_EXAMPLES:
                            with gr.Accordion("Try an example", open=False):
                                scene_gallery = gr.Gallery(
                                    value=[(row[0], Path(row[0]).stem.replace("_", " ").title()) for row in I2V_EXAMPLES],
                                    columns=3, rows=2, height=240, object_fit="cover", preview=False,
                                    label="Example images", show_label=False)
                                scene_selection = scene_gallery.select(
                                    choose_scene_example, inputs=None,
                                    outputs=[i2v_image, i2v_prompt, i2v_scene_token], queue=False,
                                )
                    with gr.Column(scale=2, min_width=390, elem_id="gae-output-panel"):
                        i2v_progressive = gr.Video(label="Your video · RGB and 3D", autoplay=True, loop=True, height=None, elem_id="gae-paired-result")
                        i2v_status = gr.Markdown()
                        gr.HTML('<div id="gae-client-latency" aria-live="polite"></div>'
                                '<progress id="gae-request-progress" aria-label="Estimated generation progress" '
                                'max="100" value="0" hidden style="width:100%;height:10px"></progress>')
                        i2v_geometry = gr.State()
                        with gr.Accordion("Downloads & details", open=False):
                            i2v_video = gr.File(label="RGB video download")
                            i2v_depth_video = gr.Video(label="Decoded depth video", autoplay=False, loop=False, height=240)
                            i2v_path = gr.State()
                            i2v_depth = gr.State()
                            with gr.Row():
                                i2v_cloud = gr.File(label="Sampled scene point cloud (.ply)")
                                i2v_download = gr.Button("Prepare point-cloud download")
                        i2v_download.click(download_pointcloud, inputs=[i2v_geometry], outputs=[i2v_cloud])
                # `.input` runs only for a direct upload/paste. Gallery selection
                # already supplies its curated prompt, so it must not also start
                # the streaming captioner and overwrite the prompt repeatedly.
                direct_scene_request = i2v_image.input(
                    mark_scene_request, inputs=[i2v_image], outputs=[i2v_scene_token], queue=False,
                )
                direct_scene_request.then(
                    prepare_direct_camera, inputs=[i2v_image, i2v_scene_token],
                    outputs=[i2v_camera_editor, i2v_camera_scene], trigger_mode="always_last",
                )
                i2v_image.input(describe_image, inputs=[i2v_image], outputs=[i2v_prompt],
                                trigger_mode="always_last", concurrency_id="caption")
                if I2V_EXAMPLES:
                    scene_selection.then(
                        prepare_direct_camera, inputs=[i2v_image, i2v_scene_token],
                        outputs=[i2v_camera_editor, i2v_camera_scene],
                        trigger_mode="always_last",
                    )
                i2v_views.change(None, inputs=[i2v_views], outputs=[],
                                 js="(v) => { window.gaeCameraEditor?.setViews(v); return []; }")
                i2v_load_preset.click(
                    None, inputs=None, outputs=[], queue=False,
                    js="() => { window.gaeCameraEditor?.loadPreset?.(); return []; }",
                )
                generation = i2v_run.click(
                    generate_direct_camera,
                    js=PLAYBACK_TIMING_JS.replace(
                        "const metric = document.getElementById('gae-client-latency');",
                        "window.gaeCameraEditor?.setViews(args[3]);\n"
                        "args[8] = window.gaeCameraEditor ? window.gaeCameraEditor.serialize() : '';\n"
                        "    const metric = document.getElementById('gae-client-latency');", 1),
                    inputs=[i2v_image, i2v_prompt, i2v_trajectory, i2v_views, i2v_steps, i2v_cfg, i2v_seed, i2v_stride,
                            i2v_camera_payload, i2v_camera_scene],
                    outputs=[i2v_video, i2v_depth_video, i2v_path, i2v_depth, i2v_cloud, i2v_status, i2v_progressive, i2v_geometry],
                )
                generation.failure(None, js="""() => {
                    window.gaeCameraEditor?.setGenerating?.(false);
                    const metric = document.getElementById('gae-client-latency');
                    const bar = document.getElementById('gae-request-progress');
                    if (metric) {
                        metric.dataset.request = 'failed';
                        metric.textContent = 'Generation failed. Please check the error message.';
                    }
                    if (bar) bar.hidden = true;
                }""")
            with gr.Tab("Video reconstruction"):
                gr.Markdown(
                    "Encode an RGB video with the GAE codec and decode it back to RGB "
                    "and depth, plus a point cloud reconstructed from the predicted "
                    "depth and rays. This tab does not run Flow/DiT generation."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vae_image = gr.Video(
                            label="Input RGB video", height=300,
                        )
                        vae_run = gr.Button("Reconstruct with VAE", variant="primary")
                    with gr.Column(scale=1):
                        vae_rgb = gr.Video(label="Reconstructed RGB video", autoplay=True, loop=True, height=300)
                        vae_depth = gr.Video(label="Reconstructed depth video", autoplay=True, loop=True, height=300)
                        vae_view = gr.Model3D(label="Reconstructed point cloud",
                                              clear_color=[1.0, 1.0, 1.0, 1.0], height=360)
                        vae_cloud = gr.File(label="Reconstructed point cloud (.ply)")
                vae_status = gr.Markdown()
                if VAE_VIDEO_EXAMPLES:
                    gr.Examples(
                        examples=VAE_VIDEO_EXAMPLES,
                        inputs=[vae_image],
                        label="VAE reconstruction examples",
                        examples_per_page=max(1, len(VAE_VIDEO_EXAMPLES)),
                    )
                vae_run.click(
                    reconstruct_vae,
                    inputs=[vae_image],
                    outputs=[vae_rgb, vae_depth, vae_cloud, vae_view, vae_status],
                )
            with gr.Tab("Text to image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        t2i_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe an image…")
                        t2i_run = gr.Button("Generate image", variant="primary")
                        with gr.Accordion("Advanced settings", open=True):
                            t2i_guidance = gr.Radio(
                                label="Guidance", choices=["IG (internal guidance)", "CFG"],
                                value="IG (internal guidance)",
                                info="IG amplifies the gap to a shallow-layer prediction; CFG to the empty prompt.")
                            t2i_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=6.0, step=0.1, value=2.0,
                                                  info="IG: 0 = no guidance. CFG: 1 = no guidance.")
                            t2i_steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=5, value=25)
                            t2i_seed = gr.Number(label="Seed", value=0, precision=0)
                            t2i_stride = gr.Slider(label="Point-cloud stride", minimum=2, maximum=8, step=1, value=4)
                        if T2I_EXAMPLES:
                            gr.Examples(
                                examples=T2I_EXAMPLES,
                                inputs=[t2i_prompt],
                                label="Repository T2I examples — examples/t2i_prompts.txt",
                                examples_per_page=8,
                            )
                    with gr.Column(scale=3, min_width=420):
                        t2i_image = gr.Image(label="Generated image", height=320)
                        t2i_status = gr.Markdown()
                        with gr.Row():
                            t2i_depth = gr.Image(label="Decoded depth", height=260)
                            t2i_view = gr.Model3D(label="Point cloud", clear_color=[1.0, 1.0, 1.0, 1.0], height=260)
                        t2i_cloud = gr.File(label="Point cloud (.ply)")
                t2i_run.click(
                    generate_t2i,
                    inputs=[t2i_prompt, t2i_steps, t2i_guidance, t2i_scale, t2i_seed, t2i_stride],
                    outputs=[t2i_image, t2i_depth, t2i_cloud, t2i_view, t2i_status],
                )
        gr.Markdown(
            """
**Quick guide:** drag a camera or use WASD; Enter adds a keyframe and F chooses Final. Click any saved camera to adjust it.

The editor previews depth estimated from one image. Empty areas are unseen; the generated video may differ from this preview.
            """
        )


if __name__ == "__main__":
    raise SystemExit("Launch Camera Studio with: bash scripts/demo/run_fast_demo.sh")
