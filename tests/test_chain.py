"""动态思维链解析测试：意象抽取、分镜规划、承接链接。"""

import json

import pytest
import requests

from fakes import FakeResponse, FakeSession
from t2video import chain, settings


def reply(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def verse(original, imagery, **extra):
    item = {"original": original, "imagery": imagery, "subject": "人", "scene": "景",
            "physics": "重力", "motion": "移动"}
    item.update(extra)
    return item


class TestExtractJson:
    def test_plain(self):
        assert chain.extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        assert chain.extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_surrounding_prose(self):
        """模型常在 JSON 前后加一句解释，必须能剥出来。"""
        assert chain.extract_json('好的，结果如下：\n{"a": 1}\n以上。') == {"a": 1}

    def test_no_json_raises(self):
        with pytest.raises(chain.ChainError):
            chain.extract_json("完全没有 JSON")

    def test_broken_json_raises_with_raw(self):
        with pytest.raises(chain.ChainError) as info:
            chain.extract_json('{"a": }')
        assert info.value.raw

    def test_non_object_raises(self):
        with pytest.raises(chain.ChainError):
            chain.extract_json("[1, 2]")


class TestRequestAnalysis:
    def test_missing_key(self, monkeypatch):
        monkeypatch.setattr(settings, "DEEPSEEK_KEY", "")
        with pytest.raises(chain.ChainError):
            chain.request_analysis("白日依山尽")

    def test_network_error(self):
        session = FakeSession([requests.ConnectionError("down")])
        with pytest.raises(chain.ChainError):
            chain.request_analysis("诗", session=session, api_key="k")

    def test_http_error(self):
        session = FakeSession([FakeResponse(401, text="bad key")])
        with pytest.raises(chain.ChainError):
            chain.request_analysis("诗", session=session, api_key="k")

    def test_empty_choices(self):
        session = FakeSession([FakeResponse(200, {"choices": []})])
        with pytest.raises(chain.ChainError):
            chain.request_analysis("诗", session=session, api_key="k")

    def test_prompt_carries_poem_and_low_temperature(self):
        session = FakeSession([reply({"verses": [verse("白日依山尽", ["落日"])]})])
        chain.request_analysis("白日依山尽", session=session, api_key="k")
        body = session.calls[0][2]["json"]
        assert "白日依山尽" in body["messages"][1]["content"]
        assert body["temperature"] <= 0.2


class TestToVerses:
    def test_keeps_order_and_fields(self):
        payload = {"verses": [verse("白日依山尽", ["落日", "山"]),
                              verse("黄河入海流", ["大河"])]}
        verses = chain.to_verses(payload)
        assert [v["original"] for v in verses] == ["白日依山尽", "黄河入海流"]
        assert verses[0]["imagery"] == ["落日", "山"]

    def test_skips_entries_without_original(self):
        payload = {"verses": [{"imagery": ["x"]}, verse("黄河入海流", ["大河"])]}
        assert len(chain.to_verses(payload)) == 1

    def test_skips_non_dict(self):
        payload = {"verses": ["字符串", verse("黄河入海流", ["大河"])]}
        assert len(chain.to_verses(payload)) == 1

    def test_empty_raises(self):
        with pytest.raises(chain.ChainError):
            chain.to_verses({"verses": []})

    def test_imagery_as_string_is_split(self):
        """模型偶尔把意象数组写成顿号连接的字符串。"""
        verses = chain.to_verses({"verses": [verse("句", "落日、孤烟、大河")]})
        assert verses[0]["imagery"] == ["落日", "孤烟", "大河"]

    def test_imagery_long_phrases_dropped(self):
        verses = chain.to_verses({"verses": [verse("句", ["落日", "一句特别长的描述不该当意象"])]})
        assert verses[0]["imagery"] == ["落日"]

    def test_imagery_dedup(self):
        verses = chain.to_verses({"verses": [verse("句", ["落日", "落日", "山"])]})
        assert verses[0]["imagery"] == ["落日", "山"]

    def test_punctuation_stripped(self):
        verses = chain.to_verses({"verses": [verse("白日依山尽。", ["落日"])]})
        assert verses[0]["original"] == "白日依山尽"


class TestPlanShots:
    def test_sparse_verses_are_merged(self):
        """意象不足 2 个的短句单独成镜会得到信息量极低的画面。"""
        verses = [verse("空山", ["山"]), verse("新雨后", ["雨", "树"])]
        shots = chain.plan_shots(verses)
        assert len(shots) == 1
        assert "空山" in shots[0]["original"] and "新雨后" in shots[0]["original"]

    def test_dense_verse_is_split(self):
        """意象超过 4 个时模型会顾此失彼，拆成两镜。"""
        verses = [verse("句", ["甲", "乙", "丙", "丁", "戊", "己"])]
        shots = chain.plan_shots(verses)
        assert len(shots) == 2
        assert len(shots[0]["imagery"]) + len(shots[1]["imagery"]) == 6

    def test_normal_verses_stay_one_to_one(self):
        verses = [verse("白日依山尽", ["落日", "山", "天"]),
                  verse("黄河入海流", ["大河", "海", "浪"])]
        shots = chain.plan_shots(verses)
        assert len(shots) == 2

    def test_index_is_one_based_and_sequential(self):
        verses = [verse(f"句{i}", ["甲", "乙", "丙"]) for i in range(4)]
        shots = chain.plan_shots(verses)
        assert [s["index"] for s in shots] == [1, 2, 3, 4]

    def test_imagery_capped(self):
        shots = chain.plan_shots([verse("句", ["甲", "乙", "丙"])])
        assert len(shots[0]["imagery"]) <= chain.DENSE_IMAGERY

    def test_empty_imagery_gets_fallback(self):
        shots = chain.plan_shots([verse("白日依山尽", [])])
        assert shots[0]["imagery"]


class TestLinkContext:
    def test_first_shot_carries_nothing(self):
        shots = chain.link_context(chain.plan_shots([verse("句", ["甲", "乙", "丙"])]))
        assert shots[0]["carried"]["subject"] == ""

    def test_later_shots_carry_previous(self):
        verses = [verse("甲句", ["甲", "乙", "丙"], subject="渔翁", scene="江面"),
                  verse("乙句", ["丁", "戊", "己"], subject="书生", scene="书斋")]
        shots = chain.link_context(chain.plan_shots(verses))
        assert shots[1]["carried"]["subject"] == "渔翁"
        assert shots[1]["carried"]["scene"] == "江面"

    def test_chain_uses_only_previous_shot(self):
        """链条只回顾上一镜：带全部历史会让提示词臃肿并冲淡重点。"""
        verses = [verse(f"句{i}", ["甲", "乙", "丙"], subject=f"主体{i}") for i in range(3)]
        shots = chain.link_context(chain.plan_shots(verses))
        assert shots[2]["carried"]["subject"] == "主体1"


class TestBuildStoryboard:
    def test_end_to_end(self):
        payload = {
            "author": "王之涣", "era": "唐", "theme": "登楼望远", "background": "边塞",
            "verses": [verse("白日依山尽", ["落日", "山", "天"]),
                       verse("黄河入海流", ["大河", "海", "浪"])],
        }
        board = chain.build_storyboard("诗", "李白风", session=FakeSession([reply(payload)]), api_key="k")
        assert board["author"] == "王之涣"
        assert board["style"] == "李白风"
        assert len(board["shots"]) == 2
        assert board["shots"][0]["carried"] == {"subject": "", "scene": "", "imagery": []}

    def test_propagates_chain_error(self):
        with pytest.raises(chain.ChainError):
            chain.build_storyboard("诗", "李白风",
                                   session=FakeSession([reply("不是 JSON")]), api_key="k")
