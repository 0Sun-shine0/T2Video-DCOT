"""提示词构造测试：优先级裁剪、长度上限、修正回馈。"""

from t2video import prompt_builder, settings, verify


def make_shot(**over):
    shot = {
        "index": 1,
        "original": "白日依山尽",
        "imagery": ["落日", "群山"],
        "subject": "一位远眺的诗人",
        "scene": "黄昏的边塞山谷",
        "physics": "衣袍受重力下垂",
        "motion": "镜头缓慢向右推移",
        "carried": {},
    }
    shot.update(over)
    return shot


BOARD = {"style": "李白风"}


class TestLengthBudget:
    def test_within_limit(self):
        text = prompt_builder.image_prompt(make_shot(), BOARD)
        assert 0 < len(text) <= settings.PROMPT_LIMIT

    def test_never_truncates_mid_clause(self):
        """回归：原实现用 text[:150] 硬切，会留下"…远处有座"这种残句。"""
        long_shot = make_shot(subject="主" * 60, scene="景" * 60, physics="理" * 60)
        text = prompt_builder.image_prompt(long_shot, BOARD)
        assert len(text) <= settings.PROMPT_LIMIT
        assert not text.endswith("…")
        # 每个从句都应当是完整的，不应出现被拦腰截断的长串
        for piece in text.split(prompt_builder.SEPARATOR):
            assert len(piece) <= 60

    def test_empty_when_nothing_fits(self):
        shot = make_shot(subject="主" * 300, scene="", physics="", imagery=[], carried={})
        text = prompt_builder.image_prompt(shot, BOARD)
        assert len(text) <= settings.PROMPT_LIMIT

    def test_video_prompt_within_limit(self):
        text = prompt_builder.video_prompt(make_shot(), BOARD)
        assert 0 < len(text) <= settings.PROMPT_LIMIT


class TestPriority:
    def test_style_anchor_survives_overload(self):
        """风格锚点丢了整片调性会飘，所以它优先级最高。"""
        shot = make_shot(subject="主" * 200, scene="景" * 200)
        text = prompt_builder.image_prompt(shot, BOARD)
        assert settings.STYLE_PROMPTS["李白风"][1] in text

    def test_low_priority_clause_dropped_first(self):
        shot = make_shot(physics="物" * 100, scene="景" * 100)
        text = prompt_builder.image_prompt(shot, BOARD)
        assert "物" * 100 not in text or "景" * 100 not in text

    def test_order_is_natural_not_sorted_by_priority(self):
        text = prompt_builder.image_prompt(make_shot(), BOARD)
        assert text.index(settings.STYLE_PROMPTS["李白风"][1]) < text.index("群山")


class TestContent:
    def test_includes_imagery_joined(self):
        assert "落日、群山" in prompt_builder.image_prompt(make_shot(), BOARD)

    def test_uses_shot_physics_over_style_default(self):
        text = prompt_builder.image_prompt(make_shot(physics="自定义约束"), BOARD)
        assert "自定义约束" in text

    def test_falls_back_to_style_physics(self):
        text = prompt_builder.image_prompt(make_shot(physics=""), BOARD)
        assert settings.PHYSICS_RULES["李白风"] in text

    def test_continuity_only_when_carried(self):
        plain = prompt_builder.image_prompt(make_shot(), BOARD)
        linked = prompt_builder.image_prompt(
            make_shot(carried={"subject": "渔翁", "scene": "江面", "imagery": []}), BOARD)
        assert "延续前一镜" not in plain
        assert "延续前一镜的渔翁" in linked

    def test_unknown_style_does_not_crash(self):
        assert prompt_builder.image_prompt(make_shot(), {"style": "未定义"}) is not None

    def test_video_prompt_includes_motion(self):
        assert "镜头缓慢向右推移" in prompt_builder.video_prompt(make_shot(), BOARD)


class TestPreviewPrompt:
    def test_within_limit_and_mentions_style(self):
        text = prompt_builder.preview_prompt("王昌龄风")
        assert len(text) <= settings.PROMPT_LIMIT
        assert settings.STYLE_PROMPTS["王昌龄风"][1] in text


class TestCorrection:
    def test_maps_each_failure_kind(self):
        cases = [
            ("第1镜清晰度", "细节"),
            ("图-视频一致性", "第一帧"),
            ("运动合理性", "运动"),
            ("片段衔接", "承接"),
            ("第1镜亮度", "曝光"),
        ]
        for name, keyword in cases:
            check = verify.Check(name, "fail", 0.1, "")
            assert keyword in prompt_builder.corrective_clause([check])

    def test_empty_for_unknown_failure(self):
        assert prompt_builder.corrective_clause([verify.Check("未知项", "fail")]) == ""

    def test_empty_input(self):
        assert prompt_builder.corrective_clause([]) == ""

    def test_duplicates_removed(self):
        checks = [verify.Check("运动合理性", "fail"), verify.Check("运动合理性", "fail")]
        text = prompt_builder.corrective_clause(checks)
        assert text.count("加入可见的主体运动与镜头推移，避免整段静止") == 1

    def test_apply_correction_merges_into_physics(self):
        shot = make_shot(physics="原约束")
        revised = prompt_builder.apply_correction(shot, "补充约束")
        assert "原约束" in revised["physics"] and "补充约束" in revised["physics"]
        assert revised["revised"] == "补充约束"

    def test_apply_empty_correction_is_noop(self):
        shot = make_shot()
        assert prompt_builder.apply_correction(shot, "") is shot

    def test_corrected_prompt_reaches_limit(self):
        shot = prompt_builder.apply_correction(make_shot(), "补充" * 50)
        assert len(prompt_builder.image_prompt(shot, BOARD)) <= settings.PROMPT_LIMIT
