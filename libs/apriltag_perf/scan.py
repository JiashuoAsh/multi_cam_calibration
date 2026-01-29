from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


@dataclass(frozen=True)
class ScanLimits:
    """扫描/检测的停止条件。

    说明：
    - target_valid: 达到多少个“有效样本”就提前停止（0 表示不按该条件停止）。
    - max_total: 最多尝试多少个候选（0 表示不限制）。
    - max_seconds: 总耗时上限（0 表示不限制）。

    这三类限制会同时生效，任何一个触发都应停止继续提交新任务。
    """

    target_valid: int = 0
    max_total: int = 0
    max_seconds: float = 0.0


@dataclass(frozen=True)
class ScanOrder:
    """候选扫描顺序的配置。"""

    strategy: str = "sequential"  # sequential | random | uniform
    seed: int = 0


@dataclass
class ScanCounters:
    """用于可观测性输出的计数器。"""

    total_candidates: int = 0
    submitted: int = 0
    completed: int = 0
    valid: int = 0
    start_ts: float = 0.0
    end_ts: float = 0.0

    def start(self) -> None:
        self.start_ts = time.monotonic()

    def stop(self) -> None:
        self.end_ts = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        if self.start_ts <= 0:
            return 0.0
        end = self.end_ts if self.end_ts > 0 else time.monotonic()
        return float(max(0.0, end - self.start_ts))


def build_scan_indices(n: int, *, order: ScanOrder, max_total: int = 0) -> List[int]:
    """根据策略生成扫描索引序列。

    策略说明：
    - sequential：0..n-1
    - random：按 seed 伪随机打乱（可复现）
    - uniform：均匀抽样（按桶选样），用于覆盖全段序列

    max_total>0 时会裁剪扫描数量。
    """

    if n <= 0:
        return []

    m = int(max_total) if int(max_total) > 0 else int(n)
    m = int(min(int(n), int(m)))

    st = (order.strategy or "sequential").strip().lower()
    if st in {"seq", "sequential", "in_order"}:
        return list(range(m))

    if st in {"rand", "random", "shuffle"}:
        idx = list(range(n))
        rng = random.Random(int(order.seed))
        rng.shuffle(idx)
        return idx[:m]

    if st in {"uniform", "even", "bucket"}:
        if m >= n:
            return list(range(n))

        # 按桶均匀取样：把 [0,n) 切成 m 个桶，每桶取一个代表。
        out: List[int] = []
        for k in range(m):
            a = int((k * n) // m)
            b = int(((k + 1) * n) // m)
            if b <= a:
                b = min(n, a + 1)
            # 取桶中心更稳（比取桶首更不容易偏前段）
            mid = int((a + b - 1) // 2)
            if len(out) == 0 or out[-1] != mid:
                out.append(mid)
        return out

    raise ValueError(f"未知 scan strategy: {order.strategy}")


def should_stop(
    *,
    limits: ScanLimits,
    counters: ScanCounters,
) -> bool:
    if int(limits.target_valid) > 0 and int(counters.valid) >= int(limits.target_valid):
        return True
    if int(limits.max_total) > 0 and int(counters.completed) >= int(limits.max_total):
        return True
    if float(limits.max_seconds) > 0 and counters.elapsed_s >= float(limits.max_seconds):
        return True
    return False


def iter_scan_sequential(
    items: Sequence[TItem],
    *,
    worker_fn: Callable[[TItem], TResult],
    is_valid_fn: Callable[[TResult], bool],
    limits: ScanLimits,
    order: ScanOrder,
    stop_fn: Optional[Callable[[TResult, ScanCounters], bool]] = None,
) -> Tuple[List[TResult], ScanCounters]:
    """顺序扫描（单进程），边执行边早停。"""

    counters = ScanCounters(total_candidates=len(items))
    counters.start()

    idxs = build_scan_indices(len(items), order=order, max_total=int(limits.max_total) if int(limits.max_total) > 0 else 0)

    out: List[TResult] = []
    for i in idxs:
        if should_stop(limits=limits, counters=counters):
            break

        counters.submitted += 1
        r = worker_fn(items[i])
        counters.completed += 1
        if is_valid_fn(r):
            counters.valid += 1
        out.append(r)

        # 动态早停：用于“达到目标连通性/覆盖度”等无法仅靠 counters 表达的条件。
        if stop_fn is not None:
            try:
                if bool(stop_fn(r, counters)):
                    break
            except Exception:
                # stop_fn 不应影响主流程：异常视为“不早停”。
                pass

    counters.stop()
    return out, counters


def iter_scan_parallel_ordered(
    items: Sequence[TItem],
    *,
    worker_fn: Callable[[TItem], TResult],
    is_valid_fn: Callable[[TResult], bool],
    limits: ScanLimits,
    order: ScanOrder,
    max_workers: int,
    prefetch: int = 0,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Tuple = (),
    stop_fn: Optional[Callable[[TResult, ScanCounters], bool]] = None,
) -> Tuple[List[TResult], ScanCounters]:
    """多进程扫描（保持提交顺序输出，便于可复现的早停）。

    设计取舍：
    - 结果按“提交顺序”消费，而不是按“完成顺序”。
      这样即便在多进程下，随机/均匀采样策略也更容易做到可复现。
    - 为了让 worker 不被阻塞，我们会提前提交 prefetch 个任务。

    注意：触发早停后，已提交但未消费的任务可能仍在运行。
    我们会停止提交新任务，并尽力 cancel 未开始的 future。
    """

    from concurrent.futures import ProcessPoolExecutor

    counters = ScanCounters(total_candidates=len(items))
    counters.start()

    if int(max_workers) <= 0:
        max_workers = 1

    if int(prefetch) <= 0:
        prefetch = int(max_workers) * 2

    idxs = build_scan_indices(len(items), order=order, max_total=int(limits.max_total) if int(limits.max_total) > 0 else 0)

    out: List[TResult] = []

    # 将 idxs 转为迭代器，按需提交。
    it = iter(idxs)
    q: Deque = deque()

    with ProcessPoolExecutor(
        max_workers=int(max_workers),
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        # 初始提交
        while len(q) < int(prefetch):
            try:
                i = next(it)
            except StopIteration:
                break
            counters.submitted += 1
            q.append(ex.submit(worker_fn, items[i]))

        while len(q) > 0:
            # 早停：不再提交新任务，并尽量取消队列中未开始的任务。
            if should_stop(limits=limits, counters=counters):
                while len(q) > 0:
                    f = q.pop()
                    try:
                        f.cancel()
                    except Exception:
                        pass
                break

            f0 = q.popleft()
            r = f0.result()
            counters.completed += 1
            if is_valid_fn(r):
                counters.valid += 1
            out.append(r)

            # 动态早停：与 iter_scan_sequential 同语义，但需要额外清空队列。
            if stop_fn is not None:
                try:
                    if bool(stop_fn(r, counters)):
                        while len(q) > 0:
                            f = q.pop()
                            try:
                                f.cancel()
                            except Exception:
                                pass
                        break
                except Exception:
                    pass

            # 继续补充提交
            while len(q) < int(prefetch):
                if should_stop(limits=limits, counters=counters):
                    break
                try:
                    i = next(it)
                except StopIteration:
                    break
                counters.submitted += 1
                q.append(ex.submit(worker_fn, items[i]))

    counters.stop()
    return out, counters
