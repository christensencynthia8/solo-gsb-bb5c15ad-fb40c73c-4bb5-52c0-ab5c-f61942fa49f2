"""巡演装运乐谱册装箱核心。

背景
====
巡演装运时，要把若干份**已保存翻页方案**各自打印装订成的**乐谱册**装入
统一规格的纸箱。每份方案一册，册厚即该方案结果的页数。装箱规则：

- 每册只能**整本**放入一箱（不可拆分到多箱）；
- 每箱**至少一册**，且箱内各册页数之和不超过箱体页容量（正整数）。

优化目标（按优先级）
====================
在所有合法装箱方案中依次比较：

1. **箱数最少**；
2. 箱数相同时，**最重箱的装载页数最小**（最重的箱子尽量轻）；
3. 仍并列时，把各箱内方案 ID 排序、箱列表再排序，取**字典序最小**的分箱
   （先比第一箱的 ID 序列，再依次比后续各箱）。

求解
====
册数 n ≤ 12，用子集动态规划。``load[S]`` 为子集 S 的页数和；S 能整装进一
箱当且仅当 ``load[S] ≤ 容量``。``dp[S]`` 为把 S 装箱的最优代价三元组
``(箱数, 最重箱装载页数, 规范箱列表)``，其中规范箱列表即"各箱内 ID 排序、
箱列表再排序"后的元组，三元组按目标优先级做字典序比较：

    dp[∅] = (0, 0, ())
    dp[S] = min over 非空 T ⊆ S 且 load[T] ≤ 容量:
            ( dp[S\\T].箱数 + 1,
              max(dp[S\\T].最重箱, load[T]),
              有序合并(dp[S\\T].规范箱列表, T 的升序 ID 元组) )

S\\T 是 S 的真子集、数值必小于 S，按掩码从小到大递推即可；子集枚举总量
O(3^n)（n ≤ 12 时约 53 万）。每册都不超容量（超容量在 DP 前直接报错），
故任何子集都可按单册一箱装完，dp 必有可行值。

最优子结构：固定"最后一箱" T 时，若 S\\T 的两个装箱方案代价同长（箱数相
同）且规范列表 L1 ≤ L2，则各自并入同一箱 T 再排序后仍有
sorted(L1+[T]) ≤ sorted(L2+[T])（两列表仅在首个不同处分叉，并入同一元素
不改变分叉先后）；箱数或最重箱不同则数值部分已分出胜负。故对 S\\T 取最优
方案再并入 T，即得"以 T 收尾"的最优方案，对所有合法 T 取最小即 dp[S]。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: 一次装运的乐谱册数下限/上限（子集 DP 状态数 2^n 随 n 指数增长）
MIN_BOOKLETS = 2
MAX_BOOKLETS = 12

#: 规范箱列表中一箱的表示：箱内方案 ID 升序元组
_BoxKey = Tuple[str, ...]
#: DP 代价：(箱数, 最重箱装载页数, 规范箱列表)
_Cost = Tuple[int, int, Tuple[_BoxKey, ...]]


class ShipmentError(ValueError):
    """乐谱册装箱输入不合法：册数越界、方案 ID 重复或非法、页数非法、
    箱体页容量非正整数、单册超容量等。"""


@dataclass(frozen=True)
class Box:
    """一箱的装箱结果。"""

    box_index: int  # 箱序号（1 基，按字典序最小的规范箱列表顺序）
    plan_ids: List[str]  # 箱内方案 ID（升序）
    loaded_pages: int  # 本箱装载页数（各册页数之和）
    remaining_capacity: int  # 本箱剩余容量（箱体页容量 − 装载页数）


@dataclass(frozen=True)
class Shipment:
    """乐谱册装箱结果。"""

    box_capacity: int  # 箱体页容量
    booklet_count: int  # 乐谱册数
    box_count: int  # 总箱数（全局最少）
    total_pages: int  # 各册页数之和
    heaviest_load: int  # 最重箱装载页数（同箱数下全局最小）
    boxes: List[Box]  # 各箱（字典序最小的规范顺序）


def _validate(
    booklets: Sequence[Tuple[str, int]], box_capacity: int
) -> Tuple[List[str], List[int]]:
    """校验全部输入，返回 (方案 ID 列表, 各册页数列表)（顺序同输入）。

    :raises ShipmentError: 任一约束不满足
    """
    if not isinstance(box_capacity, int) or isinstance(box_capacity, bool):
        raise ShipmentError(f"箱体页容量必须为整数（收到 {box_capacity!r}）")
    if box_capacity < 1:
        raise ShipmentError(
            f"箱体页容量必须为正整数（实际 {box_capacity}）"
        )

    if not isinstance(booklets, Sequence) or isinstance(booklets, (str, bytes)):
        raise ShipmentError("booklets 必须是 (方案 ID, 页数) 二元组数组")
    if not MIN_BOOKLETS <= len(booklets) <= MAX_BOOKLETS:
        raise ShipmentError(
            f"乐谱册数（{len(booklets)}）必须在 {MIN_BOOKLETS}~{MAX_BOOKLETS}"
            " 之间，一次装运至少 2 册、至多 12 册"
        )

    ids: List[str] = []
    pages: List[int] = []
    for idx, entry in enumerate(booklets):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ShipmentError(
                f"第 {idx + 1} 份乐谱册必须是 (方案 ID, 页数) 二元组"
                f"（收到 {entry!r}）"
            )
        plan_id, page_count = entry
        if not isinstance(plan_id, str) or not plan_id:
            raise ShipmentError(
                f"第 {idx + 1} 个方案 ID 必须是非空字符串（收到 {plan_id!r}）"
            )
        if (
            not isinstance(page_count, int)
            or isinstance(page_count, bool)
            or page_count < 1
        ):
            raise ShipmentError(
                f"第 {idx + 1} 份乐谱册（方案 {plan_id}）页数必须为正整数"
                f"（收到 {page_count!r}）"
            )
        ids.append(plan_id)
        pages.append(page_count)

    seen: set[str] = set()
    for plan_id in ids:
        if plan_id in seen:
            raise ShipmentError(
                f"方案 ID {plan_id} 重复：每份方案只能装一册，"
                "提交的方案 ID 必须互异"
            )
        seen.add(plan_id)

    # 每册只能整本放入一箱：单册超容量时任何装箱方案都无法容纳，直接判定。
    for plan_id, page_count in zip(ids, pages):
        if page_count > box_capacity:
            raise ShipmentError(
                f"方案 {plan_id} 的乐谱册（{page_count} 页）超过箱体页容量"
                f"（{box_capacity} 页），每册只能整本放入一箱，无法装运"
            )
    return ids, pages


def pack_booklet_boxes(
    booklets: Sequence[Tuple[str, int]], box_capacity: int
) -> Shipment:
    """为巡演装运做乐谱册装箱。

    :param booklets: 2~12 份乐谱册，每份为 (方案 ID, 页数) 二元组；方案 ID
        必须互异，页数为正整数
    :param box_capacity: 箱体页容量（正整数）
    :return: 按目标优先级最优的分箱（箱数最少 → 最重箱装载页数最小 →
        各箱 ID 排序、箱列表排序后字典序最小）
    :raises ShipmentError: 册数越界、方案 ID 重复或非法、页数非法、
        箱体页容量非正整数或单册超容量
    """
    ids, pages = _validate(booklets, box_capacity)
    n = len(ids)
    full = (1 << n) - 1

    # load[S]：子集 S 的页数和（S 非空时即该箱装载页数）。
    load = [0] * (full + 1)
    for mask in range(1, full + 1):
        bit = mask & -mask
        load[mask] = load[mask ^ bit] + pages[bit.bit_length() - 1]

    # key[S]：子集 S 的箱内升序 ID 元组（规范箱列表的单箱表示）。
    key: List[_BoxKey] = [()] * (full + 1)
    for mask in range(1, full + 1):
        key[mask] = tuple(sorted(ids[j] for j in range(n) if mask >> j & 1))

    # ---- 子集 DP：dp[S] = (箱数, 最重箱装载页数, 规范箱列表) ----
    # S\T 是 S 的真子集、数值必小于 S，按掩码升序递推；每册都不超容量，
    # 任何子集都可按单册一箱装完，dp 必有可行值。先比数值部分（箱数、
    # 最重箱），并列才合并比较规范箱列表，避免为被支配候选构造列表。
    dp: List[_Cost] = [(0, 0, ())] * (full + 1)
    for mask in range(1, full + 1):
        best_bins = n + 1
        best_heaviest = 0
        best_boxes: Tuple[_BoxKey, ...] = ()
        sub = mask
        while sub:
            if load[sub] <= box_capacity:
                prev_bins, prev_heaviest, prev_boxes = dp[mask ^ sub]
                cand_bins = prev_bins + 1
                cand_heaviest = max(prev_heaviest, load[sub])
                if (cand_bins, cand_heaviest) < (best_bins, best_heaviest):
                    best_bins = cand_bins
                    best_heaviest = cand_heaviest
                    merged = list(prev_boxes)
                    insort(merged, key[sub])
                    best_boxes = tuple(merged)
                elif (cand_bins, cand_heaviest) == (best_bins, best_heaviest):
                    merged = list(prev_boxes)
                    insort(merged, key[sub])
                    if tuple(merged) < best_boxes:
                        best_boxes = tuple(merged)
            sub = (sub - 1) & mask
        dp[mask] = (best_bins, best_heaviest, best_boxes)

    box_count, heaviest_load, canonical = dp[full]
    load_of = {key[1 << j][0]: pages[j] for j in range(n)}
    boxes = [
        Box(
            box_index=index,
            plan_ids=list(box_key),
            loaded_pages=sum(load_of[plan_id] for plan_id in box_key),
            remaining_capacity=box_capacity
            - sum(load_of[plan_id] for plan_id in box_key),
        )
        for index, box_key in enumerate(canonical, start=1)
    ]

    return Shipment(
        box_capacity=box_capacity,
        booklet_count=n,
        box_count=box_count,
        total_pages=sum(pages),
        heaviest_load=heaviest_load,
        boxes=boxes,
    )
