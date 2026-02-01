from __future__ import annotations

import json

import pytest

from mcca.core.config import load_config


def test_load_config_success(tmp_path) -> None:
    cfg_path = tmp_path / "apriltag_config.json"
    cfg_path.write_text(
        json.dumps({"_comment": "ok", "camera_settings": {"camera_type": "video_stereo"}}),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert isinstance(cfg, dict)
    assert cfg["camera_settings"]["camera_type"] == "video_stereo"


def test_load_config_missing_file_raises_system_exit(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as e:
        load_config(missing)
    assert "配置文件不存在" in str(e.value)


def test_load_config_invalid_json_raises_system_exit(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        load_config(bad)
    assert "读取配置文件失败" in str(e.value)


def test_load_config_top_level_not_object_raises_system_exit(tmp_path) -> None:
    bad = tmp_path / "bad_top.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        load_config(bad)
    assert "顶层必须是 object" in str(e.value)
