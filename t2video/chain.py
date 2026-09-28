"""动态思维链解析：把一首诗拆成意象序列，再规划成可分镜生成的场景单元。

链条是"动态"的，体现在两处：

1. 分镜数不固定。一句一镜是最省事的做法，但五言短句单独成镜信息量不足、
   意象密集的长句又装不进一次生成。这里按每句的意象数量做合并与拆分，
   分镜数由诗的意象密度决定。
2. 每一镜都携带前序分镜的视觉状态。后续镜头的提示词不是独立生成的，
   而是建立在已经画出来的画面之上，这是整条链能连贯的原因。
"""

import json
import logging
import re

import requests

from . import settings

LOGGER = logging.getLogger(__name__)

# 分镜规划的经验阈值：单句意象少于 2 个与相邻句合并，超过 4 个拆成两镜。
# 这两个数是按五言/七言律诗的实际解析结果标的，换成词或现代诗需要重标。
SPARSE_IMAGERY = 2
DENSE_IMAGERY = 4

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

ANALYSIS_PROMPT = """请把下面这首诗拆成可直接用于绘画和视频生成的场景单元。

诗：
{poem}

对每一句给出：
- original：原句
- imagery：3-5 个核心意象词，必须是可以画出来的具体物象
- subject：画面主体（人或物）
- scene：环境与场景元素
- physics：这一句需要遵守的物理约束（重力、流体、光影方向等）
- motion：镜头里应当发生的运动

同时给出整首诗的 theme（主题）、background（背景，50 字内）、author、era。

只返回 JSON，不要任何解释文字，结构如下：
{{"author":"","era":"","theme":"","background":"",
"verses":[{{"original":"","imagery":[],"subject":"","scene":"",
"physics":"","motion":""}}]}}"""


class ChainError(RuntimeError):
    """解析失败。带上原始响应片段，便于判断是模型跑偏还是网络问题。"""

    def __init__(self, message, raw=""):
        self.raw = raw
        super().__init__(message)


def extract_json(content):
    """把模型输出里最外层的 JSON 对象抠出来。

    实测 deepseek-chat 有三种跑偏方式：裹 ```json 代码块、前后加解释性文字、
    字符串里出现未转义的换行。前两种在这里处理，第三种交给调用方重试。
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ChainError("响应中找不到 JSON 对象", text[:200])

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ChainError(f"JSON 解析失败: {exc}", text[start:start + 200]) from exc

    if not isinstance(parsed, dict):
        raise ChainError("解析结果不是对象", text[:200])
    return parsed


def request_analysis(poem, session=None, api_key=None, timeout=None):
    """调用 DeepSeek 做意象抽取。失败抛 ChainError，由上层决定是否重试。"""
    key = api_key if api_key is not None else settings.DEEPSEEK_KEY
    if not key:
        raise ChainError(settings.CREDENTIAL_HINT)

    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是古典诗词的画面构建助手，输出严格的 JSON。"},
            {"role": "user", "content": ANALYSIS_PROMPT.format(poem=poem)},
        ],
        # 温度压到 0.1：这一步要的是稳定复现，不是文采。
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 2000,
    }

    caller = session or requests
    try:
        response = caller.post(
            settings.DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout or settings.DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
    except requests.exceptions.RequestException as exc:
        raise ChainError(f"DeepSeek 请求失败: {exc}") from exc
    except ValueError as exc:
        raise ChainError(f"DeepSeek 响应不是 JSON: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise ChainError("DeepSeek 未返回候选结果", str(body)[:200])
    return extract_json((choices[0].get("message") or {}).get("content", ""))


def _clean_imagery(raw):
    """意象必须是短词。模型偶尔会塞进整句，这里切掉并去重。"""
    if isinstance(raw, str):
        raw = re.split(r"[、,，;；\s]+", raw)
    if not isinstance(raw, list):
        return []

    seen, cleaned = set(), []
    for item in raw:
        word = str(item).strip().strip("。，、；")
        if not word or len(word) > 8 or word in seen:
            continue
        seen.add(word)
        cleaned.append(word)
    return cleaned


def _clean_line(text):
    """归一化原句：去掉所有空白与句末标点。

    模型返回的原句偶尔带空格或句号，不归一化会让同一句在两次解析后
    看起来不同，进而影响分镜合并的判断。
    """
    return re.sub(r"\s+", "", str(text or "")).strip("。，、；！？")


def to_verses(payload):
    """把接口返回转成逐句记录，过滤掉缺原句的条目。"""
    verses = []
    for item in payload.get("verses") or []:
        if not isinstance(item, dict):
            continue
        original = _clean_line(item.get("original"))
        if not original:
            continue
        verses.append({
            "original": original,
            "imagery": _clean_imagery(item.get("imagery")),
            "subject": str(item.get("subject") or "").strip(),
            "scene": str(item.get("scene") or "").strip(),
            "physics": str(item.get("physics") or "").strip(),
            "motion": str(item.get("motion") or "").strip(),
        })
    if not verses:
        raise ChainError("解析结果里没有任何诗句", json.dumps(payload, ensure_ascii=False)[:200])
    return verses


def _merge(verses):
    """意象稀疏的相邻句并成一镜。

    典型情况是"空山"这类两字短句，单独生成会得到一张信息量极低的图。
    """
    merged = []
    buffer = None
    for verse in verses:
        if buffer is None:
            buffer = dict(verse)
            buffer["absorbed"] = [verse["original"]]
            continue

        if len(buffer["imagery"]) < SPARSE_IMAGERY:
            buffer["original"] = buffer["original"] + "，" + verse["original"]
            buffer["absorbed"].append(verse["original"])
            buffer["imagery"] = _dedupe(buffer["imagery"] + verse["imagery"])
            for field in ("subject", "scene", "physics", "motion"):
                if verse[field] and verse[field] not in buffer[field]:
                    buffer[field] = (buffer[field] + "；" + verse[field]).strip("；")
        else:
            merged.append(buffer)
            buffer = dict(verse)
            buffer["absorbed"] = [verse["original"]]

    if buffer is not None:
        merged.append(buffer)
    return merged


def _dedupe(items):
    """去重并保持原顺序。合并相邻句时意象会取并集，重复项要清掉。"""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _split(verse):
    """意象过载的句子拆成两镜，前段和后段各分一半意象。

    七言句塞五个意象时，模型会顾此失彼，画面总有一半元素缺失。
    """
    imagery = verse["imagery"]
    half = (len(imagery) + 1) // 2
    pieces = []
    for part, (chunk, tag) in enumerate([(imagery[:half], "前半"), (imagery[half:], "后半")]):
        pieces.append({
            "original": verse["original"],
            "imagery": chunk,
            "subject": verse["subject"],
            "scene": verse["scene"],
            "physics": verse["physics"],
            "motion": verse["motion"],
            "absorbed": [f"{verse['original']}（{tag}）"],
            "part": part,
        })
    return pieces


def plan_shots(verses):
    """动态规划分镜：先合并稀疏句，再拆分密集句。"""
    shots = []
    for verse in _merge(verses):
        if len(verse["imagery"]) > DENSE_IMAGERY:
            shots.extend(_split(verse))
        else:
            shots.append(verse)

    for index, shot in enumerate(shots, 1):
        shot["index"] = index
        shot["imagery"] = shot["imagery"][:DENSE_IMAGERY] or [shot["original"][:4]]
    LOGGER.info("分镜规划完成：%d 句 -> %d 镜", len(verses), len(shots))
    return shots


def link_context(shots):
    """给每一镜挂上前序分镜的视觉状态。

    只带上一镜的关键信息而不是全部历史：链条太长会把早期细节冲淡，
    而且提示词有长度上限。
    """
    previous = None
    for shot in shots:
        shot["carried"] = {
            "subject": (previous or {}).get("subject", ""),
            "scene": (previous or {}).get("scene", ""),
            "imagery": (previous or {}).get("imagery", []),
        }
        previous = shot
    return shots


def build_storyboard(poem, style, session=None, api_key=None):
    """完整解析一首诗，返回可直接进入生成流程的分镜表。"""
    payload = request_analysis(poem, session=session, api_key=api_key)
    verses = to_verses(payload)
    shots = link_context(plan_shots(verses))
    return {
        "author": str(payload.get("author") or "").strip(),
        "era": str(payload.get("era") or "").strip(),
        "theme": str(payload.get("theme") or "").strip(),
        "background": str(payload.get("background") or "").strip(),
        "style": style,
        "verses": verses,
        "shots": shots,
    }
