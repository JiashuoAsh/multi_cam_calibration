from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CacheConfig:
    """检测缓存配置。"""

    enabled: bool = True
    cache_dir: str = "cache/apriltag_detection"
    force_redetect: bool = False


def file_fingerprint(path: Path) -> Tuple[int, int]:
    """返回 (mtime_ns, size)。

    说明：
    - 用于缓存 key：当文件内容更新（mtime/size 改变）时自动失效。
    - 这里不做 hash 全文件，避免额外 IO。
    """

    st = path.stat()
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    size = int(st.st_size)
    return mtime_ns, size


def stable_hash_dict(d: Dict[str, Any]) -> str:
    """对 dict 做稳定哈希（key 排序、确保可 JSON 序列化）。"""

    payload = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    image_path: Path,
    algo_key: Dict[str, Any],
) -> str:
    """构建缓存 key（hash string）。

    key 组成：
    - 图片绝对路径（resolve 后），避免相对路径在不同 cwd 下冲突
    - 文件 fingerprint（mtime_ns/size）
    - 检测算法关键开关与参数（由调用方提供 algo_key）
    """

    p = image_path.resolve()
    mtime_ns, size = file_fingerprint(p)

    key_dict: Dict[str, Any] = {
        "path": str(p),
        "mtime_ns": int(mtime_ns),
        "size": int(size),
        "algo": algo_key,
    }
    return stable_hash_dict(key_dict)


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent), suffix=".tmp") as f:
        tmp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(dst))


class DetectionCache:
    """简单的“每张图一个文件”的检测缓存。"""

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.dir = Path(cfg.cache_dir)

    def _path_for_key(self, key: str) -> Path:
        return self.dir / f"{key}.npz"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        if (not self.cfg.enabled) or self.cfg.force_redetect:
            return None

        p = self._path_for_key(key)
        if not p.exists():
            return None

        try:
            with np.load(str(p), allow_pickle=False) as z:
                meta_json = str(z["meta"].tolist()) if "meta" in z else "{}"
                meta = json.loads(meta_json)

                status = int(z["status"].item()) if "status" in z else 0
                ids = z["ids"] if "ids" in z else None
                corners = z["corners"] if "corners" in z else None

            return {
                "meta": meta,
                "status": status,
                "ids": ids,
                "corners": corners,
            }
        except Exception:
            # 缓存损坏时直接当作 miss
            return None

    def save(
        self,
        key: str,
        *,
        ids: Optional[np.ndarray],
        corners: Optional[np.ndarray],
        status: int,
        meta: Dict[str, Any],
    ) -> None:
        if not self.cfg.enabled:
            return

        p = self._path_for_key(key)
        p.parent.mkdir(parents=True, exist_ok=True)

        # np.savez_compressed 只能写文件路径；为保证原子性，写入临时文件后 replace。
        meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        with tempfile.NamedTemporaryFile(delete=False, dir=str(p.parent), suffix=".npz") as f:
            tmp_path = Path(f.name)

        try:
            np.savez_compressed(
                str(tmp_path),
                meta=np.array(meta_json),
                status=np.int32(int(status)),
                ids=np.asarray(ids) if ids is not None else np.zeros((0, 1), dtype=np.int32),
                corners=np.asarray(corners) if corners is not None else np.zeros((0, 4, 2), dtype=np.float32),
            )
            os.replace(str(tmp_path), str(p))
        finally:
            try:
                if tmp_path.exists() and str(tmp_path) != str(p):
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
