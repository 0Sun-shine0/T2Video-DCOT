"""一致性验证测试。

每个指标都要有可复现的行为断言：静止视频必须判为无运动，
结构不同的图必须给出低 SSIM，没有打分器时语义项必须 skip 而不是编分数。
"""

import numpy as np
import pytest

from synth import ball_frames, checkerboard, solid, write_image, write_video
from t2video import settings, verify
from t2video.verify import (
    Check,
    brightness_gap,
    check_image,
    check_image_against_clip,
    check_join,
    check_motion,
    check_semantics,
    load_image,
    motion_profile,
    ssim,
)


class TestSsim:
    def test_identical_is_one(self, checker_image):
        image = load_image(checker_image)
        assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)

    def test_different_structure_is_low(self, tmp_path):
        first = load_image(write_image(tmp_path / "a.png", checkerboard(block=10)))
        second = load_image(write_image(tmp_path / "b.png", solid((0, 0, 0))))
        assert ssim(first, second) < 0.5

    def test_shape_mismatch_resized(self, tmp_path):
        big = load_image(write_image(tmp_path / "b.png", checkerboard(size=(360, 640))))
        small = load_image(write_image(tmp_path / "s.png", checkerboard(size=(180, 320))))
        assert ssim(big, small) is not None

    def test_none_returns_none(self):
        assert ssim(None, None) is None

    def test_grayscale_supported(self):
        gray = np.full((64, 64), 100, dtype=np.uint8)
        assert ssim(gray, gray) == pytest.approx(1.0, abs=1e-6)


class TestBrightnessGap:
    def test_identical_zero(self, checker_image):
        image = load_image(checker_image)
        assert brightness_gap(image, image) == pytest.approx(0.0)

    def test_dark_vs_bright_large(self, tmp_path):
        dark = load_image(write_image(tmp_path / "d.png", solid((10, 10, 10))))
        bright = load_image(write_image(tmp_path / "b.png", solid((240, 240, 240))))
        assert brightness_gap(dark, bright) > settings.VERIFY_THRESHOLDS["brightness_delta_max"]

    def test_stable_on_low_variance_images(self, tmp_path):
        """回归：直方图相关系数在低方差图像上会病态归零。
        同一张图差 2 个灰阶，亮度均值差必须仍然很小。"""
        base = load_image(write_image(tmp_path / "a.png", solid((40, 40, 40))))
        drifted = load_image(write_image(tmp_path / "b.png", solid((42, 42, 42))))
        assert brightness_gap(base, drifted) < 5.0


class TestLoadImage:
    def test_missing_returns_none(self, tmp_path):
        assert load_image(tmp_path / "nope.png") is None

    def test_reads_chinese_path(self, tmp_path):
        """回归：cv2.imread 在 Windows 上读不了含中文的路径，必须走 imdecode。"""
        path = write_image(tmp_path / "分镜图像" / "第1镜.png", checkerboard())
        assert load_image(path) is not None


class TestDetailScore:
    def test_solid_scores_zero(self):
        assert verify.detail_score(verify.gray_of(solid())) == pytest.approx(0.0, abs=0.05)

    def test_checkerboard_scores_high(self):
        assert verify.detail_score(verify.gray_of(checkerboard())) > 20.0

    def test_mostly_blank_with_small_subject_still_passes(self):
        """水墨与写意大量留白，全局方差很低，但主体清晰。
        分块统计必须让这种画面通过，否则会误杀正常作品。"""
        frame = solid((245, 245, 245))
        frame[70:110, 140:180] = checkerboard(size=(40, 40), block=4)
        assert verify.detail_score(verify.gray_of(frame)) > settings.VERIFY_THRESHOLDS["detail_min"]

    def test_smooth_gradient_counts_as_content(self):
        """平滑渐变没有边缘但也不是空白，不能判失败。"""
        from synth import gradient

        assert verify.detail_score(verify.gray_of(gradient())) > 0.0

    def test_blank_fails(self):
        assert verify.detail_score(verify.gray_of(solid())) < settings.VERIFY_THRESHOLDS["detail_min"]

    def test_tiny_image_falls_back(self):
        tiny = np.full((8, 8), 128, dtype=np.uint8)
        assert verify.detail_score(tiny, block=16) == pytest.approx(0.0, abs=1.0)


class TestCheckImage:
    def test_structured_passes(self, checker_image):
        result = check_image(checker_image)
        assert result.status == "ok"
        assert "细节强度" in result.detail

    def test_flat_fails(self, blank_image):
        result = check_image(blank_image)
        assert result.status == "fail" and "清晰度" in result.name

    def test_sparse_composition_passes(self, tmp_path):
        """留白构图通过：这是水墨题材的常态，不能判失败。"""
        frame = solid((240, 240, 240))
        frame[60:130, 120:200] = checkerboard(size=(70, 80), block=5)
        assert check_image(write_image(tmp_path / "ink.png", frame)).status == "ok"

    def test_missing_fails(self, tmp_path):
        assert check_image(tmp_path / "nope.png").status == "fail"

    def test_label_in_name(self, checker_image):
        assert "第2镜" in check_image(checker_image, "第2镜图").name


class TestMotionProfile:
    def test_static_is_frozen(self, static_video):
        profile = motion_profile(static_video)
        assert profile["frozen_ratio"] == 1.0
        assert profile["mean_flow"] < settings.VERIFY_THRESHOLDS["frozen_flow"]

    def test_moving_has_flow(self, moving_video):
        profile = motion_profile(moving_video)
        assert profile["mean_flow"] > settings.VERIFY_THRESHOLDS["frozen_flow"]
        assert profile["frozen_ratio"] < 0.5

    def test_noise_jumps_more_than_motion(self, noisy_video, moving_video):
        assert motion_profile(noisy_video)["max_jump"] > motion_profile(moving_video)["max_jump"]

    def test_missing_returns_none(self, tmp_path):
        assert motion_profile(tmp_path / "nope.mp4") is None

    def test_single_frame_returns_none(self, tmp_path):
        assert motion_profile(write_video(tmp_path / "one.mp4", [solid()])) is None


class TestCheckMotion:
    def test_static_fails(self, static_video):
        result = check_motion(static_video)
        assert result.status == "fail" and "静态" in result.detail

    def test_moving_passes(self, moving_video):
        assert check_motion(moving_video).status == "ok"

    def test_noisy_warns(self, noisy_video):
        assert check_motion(noisy_video).status == "warn"

    def test_unreadable_fails(self, tmp_path):
        assert check_motion(tmp_path / "nope.mp4").status == "fail"


class TestImageAgainstClip:
    def test_matching_passes(self, tmp_path):
        frames = ball_frames()
        image = write_image(tmp_path / "first.png", frames[0])
        clip = write_video(tmp_path / "clip.mp4", frames)
        result = check_image_against_clip(image, clip)
        assert result.status == "ok"
        assert result.score > 0.9
        assert "亮度均值差" in result.detail

    def test_unrelated_fails(self, tmp_path):
        clip = write_video(tmp_path / "clip.mp4", ball_frames())
        image = write_image(tmp_path / "other.png", checkerboard(block=8))
        assert check_image_against_clip(image, clip).status == "fail"

    def test_missing_video_fails(self, checker_image, tmp_path):
        assert check_image_against_clip(checker_image, tmp_path / "nope.mp4").status == "fail"


class TestCheckJoin:
    def test_similar_passes(self, tmp_path):
        frames = ball_frames()
        first = write_video(tmp_path / "a.mp4", frames)
        second = write_video(tmp_path / "b.mp4", frames)
        assert check_join(first, second).status == "ok"

    def test_very_different_warns(self, tmp_path):
        dark = write_video(tmp_path / "d.mp4", [solid((5, 5, 5))] * 6)
        bright = write_video(tmp_path / "b.mp4", [solid((250, 250, 250))] * 6)
        assert check_join(dark, bright).status == "warn"

    def test_unreadable_skipped(self, moving_video, tmp_path):
        assert check_join(moving_video, tmp_path / "nope.mp4").status == "skip"


class TestCheckSemantics:
    def test_no_scorer_is_skip_not_fabricated(self, checker_image):
        """回归：原实现把轮廓个数当物体检测、和中文关键词个数比大小，
        与语义无关却输出"文本-图像不一致"。没有打分器时必须明确 skip。"""
        result = check_semantics("白日依山尽", checker_image, scorer=None)
        assert result.status == "skip" and result.score is None

    def test_scorer_used(self, checker_image):
        result = check_semantics("诗", checker_image, scorer=lambda t, p: 0.8)
        assert result.status == "ok" and result.score == pytest.approx(0.8)

    def test_low_score_warns(self, checker_image):
        assert check_semantics("诗", checker_image, scorer=lambda t, p: 0.05).status == "warn"

    def test_broken_scorer_skipped(self, checker_image):
        def boom(text, path):
            raise RuntimeError("模型不可用")

        assert check_semantics("诗", checker_image, scorer=boom).status == "skip"


class TestVerify:
    def test_collects_checks_and_problems(self, tmp_path):
        frames = ball_frames()
        images = [write_image(tmp_path / "i1.png", frames[0]),
                  write_image(tmp_path / "i2.png", solid((20, 20, 20)))]
        clips = [write_video(tmp_path / "v1.mp4", frames),
                 write_video(tmp_path / "v2.mp4", ball_frames(count=8))]

        checks, problems = verify.verify(images, clips)
        names = [c.name for c in checks]
        assert any("清晰度" in n for n in names)
        assert "运动合理性" in names
        assert "图-视频一致性" in names
        assert "片段衔接" in names
        assert problems and all(c.status == "fail" for c in problems)

    def test_clean_pipeline_has_no_problems(self, tmp_path):
        frames = ball_frames()
        images = [write_image(tmp_path / "i.png", frames[0])]
        clips = [write_video(tmp_path / "v.mp4", frames)]
        checks, problems = verify.verify(images, clips)
        assert problems == []

    def test_final_video_checked(self, tmp_path, moving_video):
        checks, _ = verify.verify([], [], final_video=moving_video)
        assert any("运动" in c.name for c in checks)

    def test_missing_final_is_problem(self, tmp_path):
        _, problems = verify.verify([], [], final_video=tmp_path / "nope.mp4")
        assert problems

    def test_empty_pipeline(self):
        checks, problems = verify.verify([], [])
        assert checks == [] and problems == []

    def test_scorer_only_when_given(self, tmp_path):
        frames = ball_frames()
        images = [write_image(tmp_path / "i.png", frames[0])]
        without, _ = verify.verify(images, [], texts=["t"])
        with_scorer, _ = verify.verify(images, [], texts=["t"], scorer=lambda t, p: 0.9)
        assert not any("语义" in c.name for c in without)
        assert any("语义" in c.name for c in with_scorer)


class TestCheckFormatting:
    def test_score_shown_when_present(self):
        assert "0.500" in str(Check("x", "ok", 0.5, "d"))

    def test_score_hidden_when_none(self):
        assert "None" not in str(Check("x", "skip", None, "d"))
