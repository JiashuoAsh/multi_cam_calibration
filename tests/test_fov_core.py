from __future__ import annotations

import math

import pytest

from mcca.core.fov import (
    PinholeIntrinsics,
    compute_fov_deg,
    pinhole_from_step3_intrinsics_dict,
)


def test_compute_fov_deg_symmetric_center() -> None:
    intri = PinholeIntrinsics(
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    out = compute_fov_deg(intri)

    expected_h = math.degrees(2.0 * math.atan((intri.width / 2.0) / intri.fx))
    expected_v = math.degrees(2.0 * math.atan((intri.height / 2.0) / intri.fy))
    expected_d = math.degrees(
        2.0
        * math.atan(
            math.sqrt(
                ((intri.width / 2.0) / intri.fx) ** 2
                + ((intri.height / 2.0) / intri.fy) ** 2
            )
        )
    )

    assert out.hfov_deg == pytest.approx(expected_h, abs=1e-9)
    assert out.vfov_deg == pytest.approx(expected_v, abs=1e-9)
    assert out.dfov_deg == pytest.approx(expected_d, abs=1e-9)


def test_compute_fov_deg_off_center_principal_point() -> None:
    intri = PinholeIntrinsics(
        fx=500.0,
        fy=600.0,
        cx=200.0,
        cy=100.0,
        width=1280,
        height=720,
    )

    out = compute_fov_deg(intri)

    expected_h = math.degrees(
        math.atan(intri.cx / intri.fx)
        + math.atan((intri.width - intri.cx) / intri.fx)
    )
    expected_v = math.degrees(
        math.atan(intri.cy / intri.fy)
        + math.atan((intri.height - intri.cy) / intri.fy)
    )

    assert out.hfov_deg == pytest.approx(expected_h, abs=1e-9)
    assert out.vfov_deg == pytest.approx(expected_v, abs=1e-9)


def test_pinhole_from_step3_intrinsics_dict_supports_camera_matrix() -> None:
    data = {
        "camera_matrix": [
            [1000.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        "image_size": [1280, 720],
    }
    intri = pinhole_from_step3_intrinsics_dict(data)
    assert intri.fx == 1000.0
    assert intri.fy == 900.0
    assert intri.cx == 640.0
    assert intri.cy == 360.0
    assert intri.width == 1280
    assert intri.height == 720


def test_pinhole_from_step3_intrinsics_dict_supports_top_level_keys_and_override_size() -> None:
    data = {
        "fx": 800.0,
        "fy": 810.0,
        "cx": 320.0,
        "cy": 240.0,
        "image_size": [640, 480],
    }
    intri = pinhole_from_step3_intrinsics_dict(data, override_image_size=(1920, 1080))
    assert intri.width == 1920
    assert intri.height == 1080
