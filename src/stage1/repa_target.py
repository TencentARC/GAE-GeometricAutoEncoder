"""Self-REPA 对齐目标：把 Feature VAE 的 latent 对齐到一个冻结强 encoder 的特征。

参考 KlingAI "Diffusing in the Right Space" (arXiv:2606.03578)：VAE 训练期加轻量
逐 token MLP 投影头，用余弦距离对齐冻结 foundation model 的 patch 特征，降低 VIV/SEC。
推理不参与，latent 不变。

可选目标：
    target = "cradio"  → 冻结 C-RADIO-B。
    target = "dinov2"  → 冻结 DINOv2（论文 repa+DINOv2-B 设定，spatial 较强）。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _tokens_to_hw(n: int) -> Tuple[int, int]:
    """token 数 → (h, w)。方形网格（C-RADIO 固定 square 输入）优先判完全平方，
    否则取常见 DINOv2 高度（504 输入下 H 恒为 36）。"""
    s = int(math.isqrt(n))
    if s * s == n:
        return s, s
    for h_cand in (36, 32, 28, 24, 20, 18, 16):
        if n % h_cand == 0:
            return h_cand, n // h_cand
    h = int(math.isqrt(n))
    while h > 0 and n % h != 0:
        h -= 1
    return h, n // h


def _resize_tokens(fmap: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """(B, C, Ht, Wt) → (B, h*w, C)，必要时双线性插值到 latent 网格。"""
    h, w = hw
    if fmap.shape[-2:] != (h, w):
        fmap = F.interpolate(fmap.float(), size=(h, w), mode="bilinear",
                             align_corners=False)
    b, c, h, w = fmap.shape
    return fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)


def repa_cosine_loss(z_proj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Token-wise cosine alignment L_tok — paper Eq. (8).

    Aligns each projected posterior token g_eta(mu_i) with its co-located
    C-RADIO target c_i. Acts through the learned projector, so it constrains
    semantics but places no constraint on the geometry among tokens.

    z_proj / target: (B, N, D).
    """
    z = F.normalize(z_proj.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return (1.0 - (z * t).sum(dim=-1)).mean()


def repa_similarity_loss(
    student: torch.Tensor,
    target: torch.Tensor,
    *,
    max_tokens: int = 0,
) -> torch.Tensor:
    r"""Pairwise similarity distillation L_struct — paper Eq. (9)-(10).

    Matches the N x N patch-patch cosine similarity matrices of the raw
    posterior mean against frozen DINOv2 targets. Because both matrices are
    N x N, the student and teacher channel dimensions need not match, so this
    constrains the raw posterior directly and restores the relational
    structure (LDS / SRSS) that token-only alignment collapses.

    匹配 patch-patch cosine 相似度矩阵 S = \hat F \hat F^T。

    student / target: (B, N, D)，两者 D 可不同——相似度矩阵恒为 (B, N, N)，与特征维无关，
    因此可同时用于 repa_proj(mu)（D=teacher 维）与 raw mu（D=latent 维）两种空间。
    teacher 侧已 detach；对角线恒为 1 不参与梯度竞争。
    max_tokens>0 时随机子采样 token 降 O(N²) 开销（默认 0=全网格）。
    """
    z = F.normalize(student.float(), dim=-1)
    t = F.normalize(target.detach().float(), dim=-1)
    if z.shape[:2] != t.shape[:2]:
        raise ValueError(
            f"repa_similarity_loss (B, N) 不匹配: student {tuple(z.shape[:2])} "
            f"vs target {tuple(t.shape[:2])}"
        )

    n = z.shape[1]
    if max_tokens > 0 and n > max_tokens:
        idx = torch.randperm(n, device=z.device)[:max_tokens]
        z = z[:, idx, :]
        t = t[:, idx, :]
        n = max_tokens

    s_z = torch.bmm(z, z.transpose(1, 2))
    s_t = torch.bmm(t, t.transpose(1, 2))
    # 去掉对角线（恒为 1，无梯度信息）
    eye = torch.eye(n, device=z.device, dtype=z.dtype).unsqueeze(0)
    off = n * (n - 1)
    return ((s_z - s_t) * (1.0 - eye)).pow(2).sum(dim=(1, 2)).mean() / max(off, 1)


def _extract_dinov2_patch_tokens(
    hidden: torch.Tensor,
    n_patch: int,
    num_register_tokens: int = 0,
) -> torch.Tensor:
    """从 DINOv2 last_hidden_state 取 patch tokens，去掉 CLS / register tokens."""
    start = 1 + int(num_register_tokens)
    z = hidden[:, start:start + n_patch]
    if z.shape[1] != n_patch:
        logger.warning(
            "DINOv2 token 数不匹配: expected %d, got %d (seq=%d, reg=%d); "
            "fallback 到 out[:, -n_patch:]",
            n_patch, z.shape[1], hidden.shape[1], num_register_tokens,
        )
        z = hidden[:, -n_patch:]
    return z


def spatial_normalize(x: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """iREPA spatial normalization（当前默认禁用，保留供实验）."""
    mu = x.mean(dim=1, keepdim=True)
    return x - gamma * mu


class CRadioTarget(nn.Module):
    """加载冻结的 NVIDIA C-RADIO，在输入图（[0,1]）上重新提 spatial 特征。

    C-RADIO 自带 input_conditioner，forward 期望 [0,1] 输入并内部做归一化。
    输入会被 resize 到 input_res（须能被 patch size 整除），输出网格再插值到 latent。
    """

    def __init__(
        self,
        version: str = "radio_v2.5-b",
        hub_repo: str = "NVlabs/RADIO",
        hub_source: str = "github",      # "github" | "local"
        input_res: int = 256,
    ):
        super().__init__()

        # DDP 下 torch.hub.load 会下载/解压到 TORCH_HOME（每节点本地 /local-ssd）；
        # 同节点多 rank 同时下载会互相覆盖、解压到一半（Directory not empty / 缺
        # hubconf.py / 部分文件）。因此每个节点只让 local_rank 0 先下载填好本地
        # cache，全局 barrier 后其它 rank 再从 cache 读。用 LOCAL_RANK（非全局 rank）
        # 才能保证多节点时每个节点各自下载到自己的 /local-ssd。
        import os
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_dist = dist.is_available() and dist.is_initialized()

        def _load():
            return torch.hub.load(
                hub_repo, "radio_model", version=version,
                progress=(local_rank == 0), skip_validation=True, source=hub_source,
            )

        try:
            if local_rank == 0:
                # 清掉之前并发下载损坏的 cache（解压到一半、缺 hubconf.py），否则
                # torch.hub 会复用损坏目录而不重新解压 → 一直失败。
                if hub_source == "github":
                    import shutil
                    hub_dir = torch.hub.get_dir()
                    repo_dir = os.path.join(hub_dir, hub_repo.replace("/", "_") + "_main")
                    if os.path.isdir(repo_dir) and not os.path.exists(
                        os.path.join(repo_dir, "hubconf.py")
                    ):
                        shutil.rmtree(repo_dir, ignore_errors=True)
                self.model = _load()
            if use_dist:
                dist.barrier()
            if local_rank != 0:
                self.model = _load()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"加载 C-RADIO 失败 (repo={hub_repo}, version={version}, "
                f"source={hub_source})。若离线，请把 RADIO repo 克隆到本地并设 "
                f"repa.cradio_hub_repo=<本地路径>、repa.cradio_hub_source=local。"
                f"若是并发下载损坏了 cache，请先删除 $TORCH_HOME/hub/NVlabs_RADIO_main "
                f"及 NVlabs-RADIO-* 后重试。原始错误: {e}"
            ) from e
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.input_res = int(input_res)
        self._dim: Optional[int] = None

    @property
    def target_dim(self) -> Optional[int]:
        return self._dim

    @torch.no_grad()
    def features(
        self,
        gt_flat01: torch.Tensor,                       # (BV, 3, H, W) in [0,1]
        all_feats: Dict[int, torch.Tensor],            # 未使用
        latent_hw: Tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(gt_flat01.float(), size=(self.input_res, self.input_res),
                          mode="bilinear", align_corners=False)
        out = self.model(x)
        spatial = out[1] if isinstance(out, (tuple, list)) else out
        bv, n, c = spatial.shape
        self._dim = c
        h0, w0 = _tokens_to_hw(n)
        fmap = spatial.transpose(1, 2).reshape(bv, c, h0, w0)
        return _resize_tokens(fmap, latent_hw)


class DINOv2Target(nn.Module):
    """冻结 DINOv2 (transformers AutoModel)，在 [0,1] 输入上提 patch 特征。

    与 eval_compare_encoders.encode_dinov2 / KlingAI metrics 一致：ImageNet norm，
    input_res 须被 patch(14) 整除。252→18×18 网格，与 DA3 latent 18×18 对齐。
    """

    def __init__(self, model_id: str = "facebook/dinov2-large", input_res: int = 252):
        super().__init__()
        from transformers import AutoModel
        import os
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_dist = dist.is_available() and dist.is_initialized()

        def _load():
            return AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)

        # local_rank 0 先下载填 HF cache，barrier 后其它 rank 读 cache（同 CRadio 策略）
        if local_rank == 0:
            self.model = _load()
        if use_dist:
            dist.barrier()
        if local_rank != 0:
            self.model = _load()

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = int(getattr(self.model.config, "patch_size", 14))
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.input_res = int(input_res)
        self.register_buffer("_mean", _IMG_MEAN.clone())
        self.register_buffer("_std", _IMG_STD.clone())
        self._dim: Optional[int] = None

    @property
    def target_dim(self) -> Optional[int]:
        return self._dim

    @torch.no_grad()
    def features(
        self,
        gt_flat01: torch.Tensor,                       # (BV, 3, H, W) in [0,1]
        all_feats: Dict[int, torch.Tensor],            # 未使用
        latent_hw: Tuple[int, int],
    ) -> torch.Tensor:
        x = gt_flat01.float()
        if x.shape[-1] != self.input_res or x.shape[-2] != self.input_res:
            x = F.interpolate(x, size=(self.input_res, self.input_res),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        h = w = x.shape[-1] // self.patch
        n_patch = h * w
        out = self.model(pixel_values=x).last_hidden_state
        z = _extract_dinov2_patch_tokens(out, n_patch, self.num_register_tokens)
        bv, _, c = z.shape
        self._dim = c
        fmap = z.transpose(1, 2).reshape(bv, c, h, w)
        return _resize_tokens(fmap, latent_hw)


class _SpatialNormWrapper(nn.Module):
    """对任意 target 输出施加 iREPA spatial normalization（论文：只对 external
    representation 做）。放在 target 侧，train/val 调用无需改动即自动生效。"""

    def __init__(self, inner: nn.Module, gamma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.inner = inner
        self.gamma = float(gamma)
        self.eps = float(eps)

    @property
    def target_dim(self) -> Optional[int]:
        return self.inner.target_dim

    @torch.no_grad()
    def features(self, gt_flat01, all_feats, latent_hw) -> torch.Tensor:
        out = self.inner.features(gt_flat01, all_feats, latent_hw)
        return spatial_normalize(out, self.gamma, self.eps)


def build_repa_target(cfg: dict, device, *, target_name: Optional[str] = None) -> nn.Module:
    """按 config 构建 REPA 目标（标准 self-REPA：MLP 投影在 VAE 内，此处只建 teacher）。

    target_name 不为 None 时覆盖 cfg["target"]，用于双老师（token cosine 与
    relational struct KD 各用不同 encoder，见 repa.struct_target）。
    """
    proj_type = str(cfg.get("proj_type", "mlp")).lower()
    if proj_type not in ("", "mlp"):
        logger.warning(
            "repa.proj_type=%r 已忽略：当前仅支持标准 MLP 投影头（Diffusing-in-the-Right-Space 设定）",
            proj_type,
        )
    if bool(cfg.get("spatial_norm", False)):
        logger.warning("repa.spatial_norm 已暂时禁用（不使用 iREPA spatial normalization）")

    target = str(target_name if target_name is not None else cfg.get("target", "cradio")).lower()
    if target in ("cradio", "cradio-b", "cradio_b", "radio"):
        tgt = CRadioTarget(
            version=cfg.get("cradio_version", "radio_v2.5-b"),
            hub_repo=cfg.get("cradio_hub_repo", "NVlabs/RADIO"),
            hub_source=cfg.get("cradio_hub_source", "github"),
            input_res=int(cfg.get("cradio_input_res", 256)),
        ).to(device)
    elif target in ("dinov2", "dino", "dinov2-large"):
        tgt = DINOv2Target(
            model_id=cfg.get("dinov2_model_id", "facebook/dinov2-large"),
            input_res=int(cfg.get("dinov2_input_res", 252)),
        ).to(device)
    else:
        raise ValueError(f"未知 repa.target={target!r}，应为 'cradio' | 'dinov2'")

    return tgt