"""提示词构造：把分镜的意象装配成平台能接受的图像/视频提示词。

平台限制 150 个字符，而一镜的理想描述（风格+主体+场景+意象+物理约束+承接）
通常超过这个长度。原做法是直接截断字符串，结果经常把句子砍成半截，
模型收到"…远处有座"这种残句反而画出莫名其妙的东西。

这里改成从句装配：每个从句带优先级，超长时按优先级从低到高整条丢弃，
保留下来的每一句都是完整的。
"""

import logging

from . import settings

LOGGER = logging.getLogger(__name__)

SEPARATOR = "，"

# 优先级数字越小越先保留。风格锚点必须在，否则整片调性会飘；
# 承接信息是"链"的体现，排在中档，实在放不下才牺牲。
P_STYLE = 1
P_SUBJECT = 1
P_IMAGERY = 2
P_PHYSICS = 3
P_SCENE = 3
P_CONTINUITY = 4


def _style_text(style):
    """取风格的画面描述。风格名未登记时返回空串，由调用方决定回退。"""
    return settings.STYLE_PROMPTS.get(style, ("", ""))[1]


def _physics_text(shot, style):
    """该镜的物理约束。优先用解析阶段针对这一句给出的约束，没有才回退到风格级。"""
    return shot.get("physics") or settings.PHYSICS_RULES.get(style, "")


def _continuity_text(shot):
    """只提上一镜里真正需要延续的两项，全带上会让提示词臃肿且稀释重点。"""
    carried = shot.get("carried") or {}
    parts = []
    if carried.get("subject"):
        parts.append(f"延续前一镜的{carried['subject']}")
    if carried.get("scene"):
        parts.append(f"背景与前一镜同为{carried['scene']}")
    return "，".join(parts)


def _fit(clauses, limit):
    """按优先级丢从句直到放进 limit 个字符以内，保持剩余部分的原顺序。

    从句是不可分割的最小单位，超长时整条丢弃而不是截断字符串——
    截断会留下"远处有座"这类残句，模型收到不完整的描述反而会画错内容。
    """
    kept = list(clauses)
    while kept:
        joined = SEPARATOR.join(text for _, text in kept)
        if len(joined) <= limit:
            return joined
        lowest = max(priority for priority, _ in kept)
        for position in range(len(kept) - 1, -1, -1):
            if kept[position][0] == lowest:
                LOGGER.debug("提示词超限，丢弃从句: %s", kept[position][1][:20])
                kept.pop(position)
                break
    return ""


def image_prompt(shot, storyboard):
    """图像提示词：静态画面描述，不含运动。"""
    style = storyboard.get("style", settings.DEFAULT_STYLE)
    clauses = [
        (P_STYLE, _style_text(style)),
        (P_SUBJECT, shot.get("subject", "")),
        (P_IMAGERY, "、".join(shot.get("imagery") or [])),
        (P_SCENE, shot.get("scene", "")),
        (P_PHYSICS, _physics_text(shot, style)),
        (P_CONTINUITY, _continuity_text(shot)),
    ]
    return _fit([(p, t.strip()) for p, t in clauses if t and t.strip()], settings.PROMPT_LIMIT)


def video_prompt(shot, storyboard):
    """视频提示词：在画面描述之外补上运动和延续要求。

    注意该文本当前不会进入图生视频接口的请求体（接口只接收图像与画面比例，
    运动描述字段名尚未确认）。它作为分镜规格参与验证与修正：
    运动不足时修正要求会并入本提示词，并同时改写图像提示词以更换输入图。
    """
    style = storyboard.get("style", settings.DEFAULT_STYLE)
    motion = shot.get("motion", "")
    clauses = [
        (P_STYLE, _style_text(style)),
        (P_SUBJECT, shot.get("subject", "")),
        (P_IMAGERY, "、".join(shot.get("imagery") or [])),
        (P_PHYSICS, _physics_text(shot, style)),
        (P_SUBJECT, motion),
        (P_CONTINUITY, _continuity_text(shot)),
    ]
    return _fit([(p, t.strip()) for p, t in clauses if t and t.strip()], settings.PROMPT_LIMIT)


def preview_prompt(style):
    """风格预览图用的提示词。短，不需要承接。"""
    desc = _style_text(style)
    clauses = [(P_STYLE, desc), (P_SUBJECT, "一幅能体现该风格精髓的代表性场景")]
    return _fit([(p, t) for p, t in clauses if t], settings.PROMPT_LIMIT)


def corrective_clause(failures):
    """把验证结论翻译成下一次生成能用的修正要求。

    这是链条的反馈端：验证不是只出一份报告，而是给出可执行的改法。
    """
    fixes = []
    for item in failures:
        name = getattr(item, "name", "")
        if "清晰度" in name:
            fixes.append("画面增加明确的主体细节与纹理，避免大面纯色")
        elif "图-视频" in name:
            fixes.append("严格以输入图为第一帧，不改变构图与主体位置")
        elif "运动" in name:
            fixes.append("加入可见的主体运动与镜头推移，避免整段静止")
        elif "衔接" in name:
            fixes.append("开场画面承接上一镜结尾的构图与光照")
        elif "亮度" in name:
            fixes.append("保持与前镜一致的曝光与色温")
    if not fixes:
        return ""
    return "；".join(dict.fromkeys(fixes))


def apply_correction(shot, correction):
    """把修正要求并入分镜，供重新生成使用。"""
    if not correction:
        return shot
    revised = dict(shot)
    revised["physics"] = (shot.get("physics", "") + "；" + correction).strip("；")
    revised["revised"] = correction
    return revised
