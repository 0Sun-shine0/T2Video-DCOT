"""测试公共设施。"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (ROOT, HERE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from synth import (  # noqa: E402
    ball_frames,
    checkerboard,
    gradient,
    noisy_frames,
    solid,
    static_frames,
    write_image,
    write_video,
)


@pytest.fixture
def checker_image(tmp_path):
    return write_image(tmp_path / "checker.png", checkerboard())


@pytest.fixture
def blank_image(tmp_path):
    return write_image(tmp_path / "blank.png", solid())


@pytest.fixture
def moving_video(tmp_path):
    return write_video(tmp_path / "moving.mp4", ball_frames())


@pytest.fixture
def static_video(tmp_path):
    return write_video(tmp_path / "static.mp4", static_frames())


@pytest.fixture
def noisy_video(tmp_path):
    return write_video(tmp_path / "noisy.mp4", noisy_frames())


@pytest.fixture
def sleeps():
    """记录退避时长，测试里不会真的睡。"""
    recorded = []

    def record(seconds):
        recorded.append(seconds)

    return record, recorded


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    """默认给假密钥，避免误打真实接口。"""
    import t2video.settings as settings

    monkeypatch.setattr(settings, "ACCESS_KEY", "test-ak")
    monkeypatch.setattr(settings, "SECRET_KEY", "test-sk")
