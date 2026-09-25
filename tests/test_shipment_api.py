"""巡演装运乐谱册装箱接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client, measure_count) -> str:
    """创建并保存一个规划：每小节高度 1、容量 1，每小节一页（页数=小节数）。"""
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1] * measure_count,
            "silence_seconds": [5] * measure_count,
            "page_height": 1,
            "min_turn_time": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _pack(client, plan_ids, box_capacity):
    return client.post(
        "/api/v1/shipments/booklet-boxes",
        json={"plan_ids": plan_ids, "box_capacity": box_capacity},
    )


def test_pack_single_box_when_all_fit(client):
    ids = [_create_plan(client, 3), _create_plan(client, 4), _create_plan(client, 2)]
    resp = _pack(client, ids, 10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["booklet_count"] == 3
    assert body["box_count"] == 1
    assert body["total_pages"] == 9
    assert body["heaviest_load"] == 9
    assert len(body["boxes"]) == 1
    box = body["boxes"][0]
    assert box["box_index"] == 1
    assert box["plan_ids"] == sorted(ids)
    assert box["loaded_pages"] == 9
    assert box["remaining_capacity"] == 1


def test_pack_min_boxes_then_heaviest_then_lexicographic(client):
    # 页数 1,2,3,4、容量 6：最优为 {1,4}+{2,3}（2 箱、最重 5）
    ids = [
        _create_plan(client, 1),
        _create_plan(client, 2),
        _create_plan(client, 3),
        _create_plan(client, 4),
    ]
    resp = _pack(client, ids, 6)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["box_count"] == 2
    assert body["heaviest_load"] == 5
    pages_of = dict(zip(ids, [1, 2, 3, 4]))
    groups = [set(box["plan_ids"]) for box in body["boxes"]]
    assert {frozenset(g) for g in groups} == {
        frozenset({ids[0], ids[3]}),
        frozenset({ids[1], ids[2]}),
    }
    # 规范形式：箱内 ID 升序、箱列表升序、装载/剩余容量正确
    box_keys = []
    for index, box in enumerate(body["boxes"], start=1):
        assert box["box_index"] == index
        assert box["plan_ids"] == sorted(box["plan_ids"])
        expected_load = sum(pages_of[pid] for pid in box["plan_ids"])
        assert box["loaded_pages"] == expected_load
        assert box["remaining_capacity"] == 6 - expected_load
        box_keys.append(tuple(box["plan_ids"]))
    assert box_keys == sorted(box_keys)


def test_pack_request_echoed_back(client):
    ids = [_create_plan(client, 2), _create_plan(client, 3)]
    payload = {"plan_ids": ids, "box_capacity": 5}
    resp = client.post("/api/v1/shipments/booklet-boxes", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["request"] == payload


def test_pack_plan_not_found_returns_404(client):
    existing = _create_plan(client, 2)
    resp = _pack(client, [existing, "does-not-exist"], 10)
    assert resp.status_code == 404
    assert "规划结果不存在" in resp.json()["detail"]
    assert "does-not-exist" in resp.json()["detail"]


def test_pack_single_booklet_over_capacity_returns_422(client):
    small = _create_plan(client, 3)
    large = _create_plan(client, 11)
    resp = _pack(client, [small, large], 10)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "超过箱体页容量" in detail
    assert "整本" in detail
    assert large in detail


@pytest.mark.parametrize(
    "payload,fragment",
    [
        # 册数越界
        ({"plan_ids": ["only-one"], "box_capacity": 10}, "2~12"),
        (
            {"plan_ids": [f"p{i:02d}" for i in range(13)], "box_capacity": 100},
            "2~12",
        ),
        # 方案 ID 重复
        ({"plan_ids": ["dup", "dup"], "box_capacity": 10}, "互异"),
        # 方案 ID 非字符串
        ({"plan_ids": ["a", 1], "box_capacity": 10}, "字符串"),
        ({"plan_ids": ["a", True], "box_capacity": 10}, "字符串"),
        ({"plan_ids": ["a", None], "box_capacity": 10}, "字符串"),
        # 箱体页容量非法
        ({"plan_ids": ["a", "b"], "box_capacity": 0}, "正整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": -3}, "正整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": 1.5}, "整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": 10.0}, "整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": True}, "整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": "10"}, "整数"),
        ({"plan_ids": ["a", "b"], "box_capacity": None}, "整数"),
    ],
)
def test_pack_invalid_payloads_return_422(client, payload, fragment):
    resp = client.post("/api/v1/shipments/booklet-boxes", json=payload)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


def test_pack_missing_and_extra_fields_rejected(client):
    ids = [_create_plan(client, 2), _create_plan(client, 3)]
    missing = client.post(
        "/api/v1/shipments/booklet-boxes", json={"plan_ids": ids}
    )
    assert missing.status_code == 422
    extra = client.post(
        "/api/v1/shipments/booklet-boxes",
        json={"plan_ids": ids, "box_capacity": 5, "notes": "x"},
    )
    assert extra.status_code == 422


def test_pack_does_not_write_new_plans(client, tmp_path):
    ids = [_create_plan(client, 2), _create_plan(client, 3), _create_plan(client, 4)]
    before = [client.get(f"/api/v1/plans/{pid}").json() for pid in ids]
    db_path = tmp_path / "data" / "plans.db"
    with sqlite3.connect(db_path) as conn:
        rows_before = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]

    resp = _pack(client, ids, 6)
    assert resp.status_code == 200, resp.text

    after = [client.get(f"/api/v1/plans/{pid}").json() for pid in ids]
    assert before == after
    with sqlite3.connect(db_path) as conn:
        rows_after = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    assert rows_after == rows_before == len(ids)


def test_pack_twelve_booklets_boundary_accepted(client):
    ids = [_create_plan(client, 1) for _ in range(12)]
    resp = _pack(client, ids, 12)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["booklet_count"] == 12
    assert body["box_count"] == 1
    assert body["boxes"][0]["plan_ids"] == sorted(ids)
