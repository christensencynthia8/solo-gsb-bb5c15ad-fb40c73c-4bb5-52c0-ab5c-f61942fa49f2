"""巡演装运乐谱册装箱核心的单元测试。"""

from __future__ import annotations

import random

import pytest

from app.shipment import (
    MAX_BOOKLETS,
    MIN_BOOKLETS,
    ShipmentError,
    pack_booklet_boxes,
)


def _brute_force(booklets, capacity):
    """枚举全部集合划分，按 (箱数, 最重箱装载, 规范箱列表) 取最优。

    返回 (箱数, 最重箱装载页数, 规范箱列表)；规范箱列表即各箱内 ID 排序、
    箱列表再排序后的元组。每册都不超容量时必有可行划分。
    """
    ids = [b[0] for b in booklets]
    pages = [b[1] for b in booklets]
    n = len(ids)
    best = None
    assign = [0] * n

    def rec(i, nblocks):
        nonlocal best
        if i == n:
            loads = [0] * nblocks
            for j, block in enumerate(assign):
                loads[block] += pages[j]
            if any(load > capacity for load in loads):
                return
            boxes = tuple(
                sorted(
                    tuple(
                        sorted(
                            ids[j] for j in range(n) if assign[j] == block
                        )
                    )
                    for block in range(nblocks)
                )
            )
            key = (nblocks, max(loads), boxes)
            if best is None or key < best:
                best = key
            return
        for block in range(nblocks + 1):
            assign[i] = block
            rec(i + 1, nblocks + (block == nblocks))

    rec(0, 0)
    assert best is not None
    return best


def _check_invariants(result, booklets, capacity):
    """通用结构与装箱不变量检查。"""
    ids = [b[0] for b in booklets]
    pages = dict(booklets)
    assert result.box_capacity == capacity
    assert result.booklet_count == len(booklets)
    assert result.total_pages == sum(pages.values())
    assert result.box_count == len(result.boxes)
    # 每册恰好入一箱，箱内 ID 升序，箱列表升序
    packed = []
    box_keys = []
    for index, box in enumerate(result.boxes, start=1):
        assert box.box_index == index
        assert box.plan_ids == sorted(box.plan_ids)
        assert box.loaded_pages == sum(pages[pid] for pid in box.plan_ids)
        assert 1 <= len(box.plan_ids) <= len(booklets)
        assert box.loaded_pages <= capacity
        assert box.remaining_capacity == capacity - box.loaded_pages
        packed.extend(box.plan_ids)
        box_keys.append(tuple(box.plan_ids))
    assert sorted(packed) == sorted(ids)
    assert len(set(packed)) == len(ids)
    assert box_keys == sorted(box_keys)
    assert result.heaviest_load == max(b.loaded_pages for b in result.boxes)


# ---------------------------------------------------------------- 确定性用例


def test_single_box_when_all_fit():
    # 总页数不超容量：一箱装完
    result = pack_booklet_boxes([("a", 3), ("b", 4), ("c", 2)], 10)
    assert result.box_count == 1
    assert result.total_pages == 9
    assert result.heaviest_load == 9
    box = result.boxes[0]
    assert box.plan_ids == ["a", "b", "c"]
    assert box.loaded_pages == 9
    assert box.remaining_capacity == 1
    _check_invariants(result, [("a", 3), ("b", 4), ("c", 2)], 10)


def test_min_box_count_first_priority():
    # 4 册各 3 页、容量 6：两箱各 [3,3]；箱数优先于最重箱
    booklets = [("a", 3), ("b", 3), ("c", 3), ("d", 3)]
    result = pack_booklet_boxes(booklets, 6)
    assert result.box_count == 2
    assert result.heaviest_load == 6
    assert [box.loaded_pages for box in result.boxes] == [6, 6]
    _check_invariants(result, booklets, 6)


def test_heaviest_box_minimized_given_box_count():
    # 页数 1,2,3,4、容量 6：两箱可行中 {1,4}+{2,3} 最重箱 5 最小
    booklets = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    result = pack_booklet_boxes(booklets, 6)
    assert result.box_count == 2
    assert result.heaviest_load == 5
    assert [tuple(b.plan_ids) for b in result.boxes] == [("a", "d"), ("b", "c")]
    assert [b.loaded_pages for b in result.boxes] == [5, 5]
    _check_invariants(result, booklets, 6)


def test_lexicographic_tie_break_among_equal_loads():
    # 4 册各 2 页、容量 4：三种两两配对同为 (2 箱, 最重 4)，
    # 规范箱列表 (a,b),(c,d) 字典序最小
    booklets = [("d", 2), ("b", 2), ("a", 2), ("c", 2)]
    result = pack_booklet_boxes(booklets, 4)
    assert [tuple(b.plan_ids) for b in result.boxes] == [("a", "b"), ("c", "d")]
    _check_invariants(result, booklets, 4)


def test_lexicographic_tie_break_first_box_decides():
    # 页数 1,1,2,2、容量 4：最优 (2 箱, 最重 3) 有 {a,c}+{b,d} 与
    # {a,d}+{b,c} 两种，首箱 (a,c) < (a,d)，取前者
    booklets = [("a", 1), ("b", 1), ("c", 2), ("d", 2)]
    result = pack_booklet_boxes(booklets, 4)
    assert result.heaviest_load == 3
    assert [tuple(b.plan_ids) for b in result.boxes] == [("a", "c"), ("b", "d")]
    _check_invariants(result, booklets, 4)


def test_lexicographic_prefix_tie_prefers_shorter_box():
    # 3 册各 1 页、容量 2：最优 (2 箱, 最重 2) 有 {a,b}+{c}、{a,c}+{b}、
    # {b,c}+{a} 三种，首箱 (a,) 是 (a,b)/(a,c) 的前缀且更短，取 {a}+{b,c}
    booklets = [("a", 1), ("b", 1), ("c", 1)]
    result = pack_booklet_boxes(booklets, 2)
    assert [tuple(b.plan_ids) for b in result.boxes] == [("a",), ("b", "c")]
    _check_invariants(result, booklets, 2)


def test_each_box_at_least_one_booklet():
    # 3 册各 1 页、容量 3：一箱即可装完，不存在空箱
    booklets = [("a", 1), ("b", 1), ("c", 1)]
    result = pack_booklet_boxes(booklets, 3)
    assert result.box_count == 1
    assert all(len(b.plan_ids) >= 1 for b in result.boxes)


def test_each_booklet_in_own_box_when_capacity_tight():
    booklets = [("a", 4), ("b", 5), ("c", 3)]
    result = pack_booklet_boxes(booklets, 5)
    assert result.box_count == 3
    assert [tuple(b.plan_ids) for b in result.boxes] == [("a",), ("b",), ("c",)]
    assert [b.remaining_capacity for b in result.boxes] == [1, 0, 2]
    _check_invariants(result, booklets, 5)


def test_string_id_ordering_not_input_order():
    # 输入顺序打乱不影响规范输出（按 ID 字符串排序）
    booklets = [("plan-c", 2), ("plan-a", 2), ("plan-b", 2)]
    result = pack_booklet_boxes(booklets, 2)
    assert [tuple(b.plan_ids) for b in result.boxes] == [
        ("plan-a",),
        ("plan-b",),
        ("plan-c",),
    ]
    _check_invariants(result, booklets, 2)


# ---------------------------------------------------------------- 随机对照


@pytest.mark.parametrize("seed", range(300))
def test_random_instances_match_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(MIN_BOOKLETS, 8)
    pages = [rng.randint(1, 12) for _ in range(n)]
    # 容量至少容纳最大单册（保证有解），覆盖"必须多箱"与"可装一箱"两类
    capacity = rng.randint(max(pages), max(pages) + 12)
    booklets = [(f"plan-{i:02d}", p) for i, p in enumerate(pages)]
    rng.shuffle(booklets)

    result = pack_booklet_boxes(booklets, capacity)
    ref_boxes, ref_heaviest, ref_canonical = None, None, None
    ref = _brute_force(booklets, capacity)
    ref_boxes, ref_heaviest, ref_canonical = ref

    assert result.box_count == ref_boxes
    assert result.heaviest_load == ref_heaviest
    assert [tuple(b.plan_ids) for b in result.boxes] == list(ref_canonical)
    _check_invariants(result, booklets, capacity)


def test_twelve_booklets_worst_case_performance():
    # 12 册（上限）随机页数：O(3^n) 子集 DP 应快速完成
    rng = random.Random(20260925)
    booklets = [(f"plan-{i:02d}", rng.randint(1, 20)) for i in range(12)]
    capacity = 25
    result = pack_booklet_boxes(booklets, capacity)
    _check_invariants(result, booklets, capacity)
    ref = _brute_force(booklets, capacity)
    assert result.box_count == ref[0]
    assert result.heaviest_load == ref[1]
    assert [tuple(b.plan_ids) for b in result.boxes] == list(ref[2])


def test_max_booklets_boundary_accepted():
    booklets = [(f"p{i:02d}", 1) for i in range(MAX_BOOKLETS)]
    result = pack_booklet_boxes(booklets, MAX_BOOKLETS)
    assert result.box_count == 1
    assert result.boxes[0].plan_ids == [f"p{i:02d}" for i in range(MAX_BOOKLETS)]


# ---------------------------------------------------------------- 输入校验


def test_too_few_booklets_rejected():
    with pytest.raises(ShipmentError, match="2~12"):
        pack_booklet_boxes([("a", 1)], 10)


def test_too_many_booklets_rejected():
    booklets = [(f"p{i:02d}", 1) for i in range(MAX_BOOKLETS + 1)]
    with pytest.raises(ShipmentError, match="2~12"):
        pack_booklet_boxes(booklets, 100)


def test_duplicate_plan_ids_rejected():
    with pytest.raises(ShipmentError, match="重复"):
        pack_booklet_boxes([("a", 1), ("b", 2), ("a", 1)], 10)


@pytest.mark.parametrize("capacity", [0, -1, -100])
def test_non_positive_capacity_rejected(capacity):
    with pytest.raises(ShipmentError, match="正整数"):
        pack_booklet_boxes([("a", 1), ("b", 2)], capacity)


@pytest.mark.parametrize("capacity", [1.0, 1.5, True, False, "10", None])
def test_non_integer_capacity_rejected(capacity):
    with pytest.raises(ShipmentError, match="整数"):
        pack_booklet_boxes([("a", 1), ("b", 2)], capacity)


def test_single_booklet_over_capacity_rejected():
    with pytest.raises(ShipmentError) as exc:
        pack_booklet_boxes([("a", 5), ("b", 11), ("c", 3)], 10)
    message = str(exc.value)
    assert "超过箱体页容量" in message
    assert "整本" in message
    assert "b" in message


def test_first_oversized_booklet_reported():
    # 多册超容量时报错信息指向其中一册即可，但不得静默通过
    with pytest.raises(ShipmentError, match="超过箱体页容量"):
        pack_booklet_boxes([("a", 20), ("b", 30)], 10)


@pytest.mark.parametrize(
    "booklets",
    [
        [("a", 1), ("b", 0)],  # 页数为 0
        [("a", 1), ("b", -2)],  # 页数为负
        [("a", 1), ("b", 1.5)],  # 页数为小数
        [("a", 1), ("b", True)],  # 页数为布尔
        [("a", 1), ("b", "3")],  # 页数为字符串
        [("a", 1), ("", 2)],  # 空方案 ID
        [("a", 1), (None, 2)],  # 方案 ID 非字符串
        [("a", 1), ("b",)],  # 二元组长度不足
        [("a", 1), "b"],  # 非二元组
    ],
)
def test_invalid_booklet_entries_rejected(booklets):
    with pytest.raises(ShipmentError):
        pack_booklet_boxes(booklets, 10)
