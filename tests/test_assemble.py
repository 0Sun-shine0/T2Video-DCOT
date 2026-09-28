"""素材落盘与成片合成测试。"""

import shutil

import pytest
import requests

from fakes import FakeResponse, FakeSession
from t2video import assemble
from t2video.assemble import fetch, merge, safe_name, sweep_temp


class TestSafeName:
    def test_strips_illegal_chars(self):
        assert safe_name('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"

    def test_keeps_chinese(self):
        assert safe_name("静夜思") == "静夜思"

    def test_reserved_names_escaped(self):
        assert safe_name("CON") == "_CON"
        assert safe_name("com1") == "_com1"

    def test_empty_gets_fallback(self):
        """回归：原实现遇到全是非法字符的输入会返回空串，最后生成 ".mp4"。"""
        assert safe_name("???***") == "untitled"
        assert safe_name("") == "untitled"
        assert safe_name(None) == "untitled"

    def test_length_capped(self):
        assert len(safe_name("x" * 200, limit=20)) <= 20

    def test_whitespace_collapsed(self):
        assert safe_name("a   b") == "a_b"


class TestFetch:
    def test_writes_chunks(self, tmp_path):
        session = FakeSession([FakeResponse(200, chunks=[b"abc", b"def"])])
        target = tmp_path / "out.bin"
        assert fetch("http://x/f", target, session=session) is True
        assert target.read_bytes() == b"abcdef"

    def test_timeout_passed(self, tmp_path):
        session = FakeSession([FakeResponse(200, chunks=[b"x"])])
        fetch("http://x/f", tmp_path / "o.bin", timeout=7, session=session)
        assert session.calls[0][2]["timeout"] == 7

    def test_empty_body_fails_and_cleans(self, tmp_path):
        """回归：原实现只看有没有抛异常，200 + 空 body 也会留下 0 字节文件。"""
        session = FakeSession([FakeResponse(200, chunks=[])])
        target = tmp_path / "out.bin"
        assert fetch("http://x/f", target, session=session) is False
        assert not target.exists()

    def test_http_error_fails(self, tmp_path):
        session = FakeSession([FakeResponse(404, text="nope")])
        target = tmp_path / "out.bin"
        assert fetch("http://x/f", target, session=session) is False
        assert not target.exists()

    def test_network_error_fails(self, tmp_path):
        session = FakeSession([requests.ConnectionError("down")])
        target = tmp_path / "out.bin"
        assert fetch("http://x/f", target, session=session) is False

    def test_creates_parents(self, tmp_path):
        session = FakeSession([FakeResponse(200, chunks=[b"x"])])
        target = tmp_path / "deep" / "nested" / "o.bin"
        assert fetch("http://x/f", target, session=session) is True


class TestMerge:
    def test_merges_real_clips(self, tmp_path, moving_video, static_video, monkeypatch):
        sources = [moving_video, static_video]

        def fake_fetch(url, target, **kwargs):
            shutil.copyfile(sources[int(str(url).rsplit("/", 1)[-1]) - 1], target)
            return True

        monkeypatch.setattr(assemble, "fetch", fake_fetch)
        output = merge(["http://x/1", "http://x/2"], tmp_path / "out", "成片")

        assert output is not None
        from pathlib import Path

        assert Path(output).is_file() and Path(output).stat().st_size > 0
        assert assemble.read_frame(output, "first") is not None

    def test_empty_input(self, tmp_path):
        assert merge([], tmp_path, "x") is None

    def test_all_downloads_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "fetch", lambda *a, **k: False)
        assert merge(["http://x/1"], tmp_path, "x") is None

    def test_progress_reported(self, tmp_path, moving_video, monkeypatch):
        monkeypatch.setattr(assemble, "fetch",
                            lambda url, target, **k: shutil.copyfile(moving_video, target) or True)
        seen = []
        merge(["http://x/1", "http://x/2"], tmp_path / "out", "x",
              on_progress=lambda i, t: seen.append((i, t)))
        assert seen == [(1, 2), (2, 2)]

    def test_unparsable_clip_skipped(self, tmp_path, moving_video, monkeypatch):
        def fake_fetch(url, target, **kwargs):
            if str(url).endswith("bad"):
                target.write_bytes(b"garbage")
            else:
                shutil.copyfile(moving_video, target)
            return True

        monkeypatch.setattr(assemble, "fetch", fake_fetch)
        assert merge(["http://x/bad", "http://x/good"], tmp_path / "out", "x") is not None

    def test_temp_dir_removed(self, tmp_path, moving_video, monkeypatch):
        created = []
        original = assemble.tempfile.mkdtemp

        def tracker(prefix=None):
            path = original(prefix=prefix)
            created.append(path)
            return path

        monkeypatch.setattr(assemble.tempfile, "mkdtemp", tracker)
        monkeypatch.setattr(assemble, "fetch",
                            lambda url, target, **k: shutil.copyfile(moving_video, target) or True)
        merge(["http://x/1"], tmp_path / "out", "x")

        from pathlib import Path

        assert created and all(not Path(p).exists() for p in created)


class TestReadFrame:
    def test_first_and_last(self, moving_video):
        assert assemble.read_frame(moving_video, "first") is not None
        assert assemble.read_frame(moving_video, "last") is not None

    def test_missing_file(self, tmp_path):
        assert assemble.read_frame(tmp_path / "nope.mp4") is None


class TestSweepTemp:
    def test_removes_own_stale_dirs_only(self, tmp_path, monkeypatch):
        """回归：原实现遍历系统临时目录按时间删所有文件，会误删别的程序的东西。"""
        import os
        import time

        monkeypatch.setattr(assemble.tempfile, "gettempdir", lambda: str(tmp_path))

        foreign = tmp_path / "other_app"
        foreign.mkdir()
        keep_file = foreign / "important.txt"
        keep_file.write_text("keep", encoding="utf-8")

        own = tmp_path / f"{assemble.TEMP_PREFIX}old"
        own.mkdir()

        stale = time.time() - 48 * 3600
        os.utime(foreign, (stale, stale))
        os.utime(keep_file, (stale, stale))
        os.utime(own, (stale, stale))

        assert sweep_temp(max_age_hours=24) == 1
        assert not own.exists()
        assert keep_file.read_text(encoding="utf-8") == "keep"

    def test_keeps_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble.tempfile, "gettempdir", lambda: str(tmp_path))
        fresh = tmp_path / f"{assemble.TEMP_PREFIX}new"
        fresh.mkdir()
        assert sweep_temp() == 0
        assert fresh.exists()
