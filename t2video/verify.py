"""跨媒体一致性验证。

只报告算得出来的量。这里没有视觉语义模型，因此不做任何伪装成"物体检测"的
轮廓计数；真正需要语义对齐的地方留 scorer 钩子，没有就明确标记 skip。

指标选择上有一条实测教训值得写下来：最初用直方图相关系数衡量首帧与输入图的
一致性，结果同一张图与其压缩后视频首帧的 SSIM 高达 0.998，直方图相关系数却是
-0.0015。原因是画面色调集中在少数灰阶时直方图接近常数向量，相关系数被极小的
数值扰动主导，完全无法反映真实差异。现在改用亮度均值差，数值稳定且含义明确。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import settings
from .assemble import read_frame

LOGGER = logging.getLogger(__name__)

GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299])  # OpenCV 的 BGR 顺序
SSIM_WINDOW = (11, 11)
SSIM_SIGMA = 1.5
MOTION_WIDTH = 320  # 光流前的缩放宽度，320 足够看出运动且快得多


@dataclass
class Check:
    """单项验证的结论。score 为 None 时表示该项没有可计算的数值。"""

    name: str
    status: str                 # ok / warn / fail / skip
    score: float | None = None
    detail: str = ""

    def __str__(self):
        value = "" if self.score is None else f" [{self.score:.3f}]"
        return f"{self.name}: {self.status}{value} — {self.detail}"


def gray_of(image):
    """转灰度浮点图。已是单通道时直接返回，避免重复加权。"""
    if image is None:
        return None
    if image.ndim == 2:
        return image.astype(np.float64)
    return image[..., :3].astype(np.float64) @ GRAY_WEIGHTS


def ssim(a, b, data_range=255.0):
    """高斯窗 SSIM。numpy 实现，避免为一个指标引入 scikit-image。

    尺寸不一致时把 b 缩放到 a 的尺寸。任一输入为空返回 None。
    """
    ga, gb = gray_of(a), gray_of(b)
    if ga is None or gb is None:
        return None
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))

    # 标准 SSIM 的稳定常数，K1=0.01、K2=0.03
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    mu_a = cv2.GaussianBlur(ga, SSIM_WINDOW, SSIM_SIGMA)
    mu_b = cv2.GaussianBlur(gb, SSIM_WINDOW, SSIM_SIGMA)
    var_a = cv2.GaussianBlur(ga * ga, SSIM_WINDOW, SSIM_SIGMA) - mu_a * mu_a
    var_b = cv2.GaussianBlur(gb * gb, SSIM_WINDOW, SSIM_SIGMA) - mu_b * mu_b
    cov = cv2.GaussianBlur(ga * gb, SSIM_WINDOW, SSIM_SIGMA) - mu_a * mu_b

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(numerator / denominator))


def brightness_gap(a, b):
    """两幅图的平均亮度差。用来判断曝光或色调是否突变。

    没有采用直方图相关系数：画面色调集中在少数灰阶时直方图接近常数向量，
    相关系数会被极小的数值扰动主导，实测同一张图与其压缩后视频首帧
    结构相似度 0.998 而相关系数仅 -0.0015，无法反映真实差异。
    """
    ga, gb = gray_of(a), gray_of(b)
    if ga is None or gb is None:
        return None
    return float(abs(ga.mean() - gb.mean()))


def load_image(path):
    """读入彩色图，不存在或无法解码时返回 None。

    走 imdecode 而不是 imread：Windows 上 imread 打不开含中文的路径，
    而本程序的项目目录正是用诗句命名的。
    """
    path = Path(path)
    if not path.is_file():
        return None
    buffer = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def motion_profile(video_path, frozen_below=None):
    """Farneback 光流统计。返回 None 表示帧数不足，无法估计运动。

    解码前先缩到 MOTION_WIDTH 宽：320 像素已足够看出运动，
    全分辨率算光流要慢一个量级。
    """
    frozen_below = settings.VERIFY_THRESHOLDS["frozen_flow"] if frozen_below is None else frozen_below
    capture = cv2.VideoCapture(str(video_path))
    flows, jumps, previous = [], [], None

    try:
        if not capture.isOpened():
            return None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = gray_of(frame)
            if gray.shape[1] > MOTION_WIDTH:
                scale = MOTION_WIDTH / gray.shape[1]
                gray = cv2.resize(gray, (MOTION_WIDTH, max(1, int(gray.shape[0] * scale))))
            gray = gray.astype(np.uint8)

            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flows.append(float(magnitude.mean()))
                jumps.append(float(np.abs(gray.astype(np.int16)
                                          - previous.astype(np.int16)).mean()))
            previous = gray
    finally:
        capture.release()

    if not flows:
        return None

    flows_arr, jumps_arr = np.asarray(flows), np.asarray(jumps)
    return {
        "mean_flow": float(flows_arr.mean()),
        "frozen_ratio": float((flows_arr < frozen_below).mean()),
        "max_jump": float(jumps_arr.max()),
        "pairs": len(flows),
    }


DETAIL_BLOCK = 8
DETAIL_TOP = 0.05


def detail_score(gray, block=DETAIL_BLOCK, top=DETAIL_TOP):
    """画面细节强度：边缘能量最高的一批分块的平均值。

    不用全局拉普拉斯方差。水墨与写意本来就大量留白，整幅方差会很低，
    用全局方差会把正常作品误判成"糊"。取最强的 5% 分块则只要画面里有
    真实内容就能通过。

    参数按合成素材实测标定（block=8、top=5%）：
        纯色 0.00 ／ 平滑渐变 1.25 ／ 留白构图 56 ／ 棋盘格 91
    阈值取 1.0，正好落在"完全空白"和"有任何内容"之间。
    """
    edges = np.abs(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F))
    height, width = edges.shape
    rows, cols = height // block, width // block
    if rows < 1 or cols < 1:
        return float(edges.mean())

    tiles = edges[:rows * block, :cols * block].reshape(rows, block, cols, block)
    means = np.sort(tiles.mean(axis=(1, 3)).ravel())
    count = max(1, int(means.size * top))
    return float(means[-count:].mean())


def check_image(path, label="图像"):
    """检查单张分镜图像。先判细节，再判亮度，返回第一项命中的结论。"""
    image = load_image(path)
    if image is None:
        return Check(f"{label}可读性", "fail", None, f"无法读取 {path}")

    gray = gray_of(image)
    detail = detail_score(gray)
    level = float(gray.mean())
    low, high = settings.VERIFY_BRIGHTNESS_RANGE

    if detail < settings.VERIFY_THRESHOLDS["detail_min"]:
        return Check(f"{label}清晰度", "fail", detail,
                     f"细节强度 {detail:.1f}，画面接近纯色或整体失焦")
    if not low <= level <= high:
        return Check(f"{label}亮度", "warn", level, f"平均亮度 {level:.1f} 超出 [{low}, {high}]")
    return Check(f"{label}质量", "ok", detail, f"细节强度 {detail:.1f}，亮度 {level:.1f}")


def check_image_against_clip(image_path, clip_path):
    """比对输入图与视频首帧。结构相似度不足判不通过，曝光偏移判警告。

    输入图读不到时返回 skip 而不是 fail：缺图属于上游未产出，
    不应当作"生成质量不合格"来触发重生成。
    """
    image = load_image(image_path)
    if image is None:
        return Check("图-视频一致性", "skip", None, f"无法读取 {image_path}")

    first = read_frame(clip_path, "first")
    if first is None:
        return Check("图-视频一致性", "fail", None, f"无法读取视频首帧 {clip_path}")

    score = ssim(image, first)
    gap = brightness_gap(image, first)
    detail = f"SSIM={score:.3f}，亮度均值差={gap:.1f}"

    if score < settings.VERIFY_THRESHOLDS["ssim_min"]:
        return Check("图-视频一致性", "fail", score, f"首帧与输入图差异过大（{detail}）")
    if gap > settings.VERIFY_THRESHOLDS["brightness_delta_max"]:
        return Check("图-视频一致性", "warn", score, f"曝光或色调偏移较大（{detail}）")
    return Check("图-视频一致性", "ok", score, detail)


def check_motion(clip_path):
    """判断这一镜是否真的在动。冻结帧占比过高判不通过，跳变过大判警告。"""
    profile = motion_profile(clip_path)
    if profile is None:
        return Check("运动合理性", "fail", None, f"帧数不足，无法估计运动 {clip_path}")

    detail = (f"平均光流={profile['mean_flow']:.2f}px，"
              f"冻结帧占比={profile['frozen_ratio']:.0%}，"
              f"最大帧间跳变={profile['max_jump']:.1f}（{profile['pairs']} 对样本）")

    if profile["frozen_ratio"] > settings.VERIFY_THRESHOLDS["frozen_ratio_max"]:
        return Check("运动合理性", "fail", profile["mean_flow"],
                     f"多数相邻帧没有位移，疑似静态画面拼接。{detail}")
    if profile["max_jump"] > settings.VERIFY_THRESHOLDS["flicker_max"]:
        return Check("运动合理性", "warn", profile["mean_flow"], f"存在明显闪烁或跳帧。{detail}")
    return Check("运动合理性", "ok", profile["mean_flow"], detail)


def check_join(previous_clip, next_clip):
    """比对前段末帧与后段首帧。跳变意味着镜头切换处不连贯。"""
    tail = read_frame(previous_clip, "last")
    head = read_frame(next_clip, "first")
    if tail is None or head is None:
        return Check("片段衔接", "skip", None, "首帧或末帧不可读")

    score = ssim(tail, head)
    if score < settings.VERIFY_THRESHOLDS["clip_continuity_min"]:
        return Check("片段衔接", "warn", score, f"切换处画面跳变较大（末帧-首帧 SSIM={score:.3f}）")
    return Check("片段衔接", "ok", score, f"末帧-首帧 SSIM={score:.3f}")


def check_semantics(text, image_path, scorer=None):
    """文本与画面的语义对齐。需要多模态打分器，没有就跳过而不是编一个分数。"""
    if scorer is None:
        return Check("文本-画面语义对齐", "skip", None, "未接入多模态打分器，跳过语义检查")
    try:
        score = float(scorer(text, image_path))
    except Exception as exc:
        return Check("文本-画面语义对齐", "skip", None, f"打分器执行失败: {exc}")
    return Check("文本-画面语义对齐", "ok" if score >= 0.25 else "warn", score,
                 f"语义相似度 {score:.3f}")


def verify(image_paths, clip_paths, final_video=None, texts=None, scorer=None):
    """对整条链的产物做一次验证。

    返回 (全部结论列表, 其中判为不通过的项)；后者交给反馈环节生成修正要求。
    texts 与 scorer 需要同时提供才会做语义对齐检查。
    """
    checks = []
    image_paths = [Path(p) for p in image_paths or []]
    clip_paths = [Path(p) for p in clip_paths or []]

    for order, image_path in enumerate(image_paths, 1):
        checks.append(check_image(image_path, f"第{order}镜图"))
        if scorer is not None and texts:
            text = texts[order - 1] if order - 1 < len(texts) else ""
            checks.append(check_semantics(text, image_path, scorer))

    for order, clip_path in enumerate(clip_paths, 1):
        checks.append(check_motion(clip_path))
        if order - 1 < len(image_paths):
            checks.append(check_image_against_clip(image_paths[order - 1], clip_path))

    for order in range(1, len(clip_paths)):
        checks.append(check_join(clip_paths[order - 1], clip_paths[order]))

    if final_video:
        final = Path(final_video)
        if final.is_file():
            checks.append(check_motion(final))
        else:
            checks.append(Check("成片检查", "fail", None, f"成片不存在: {final}"))

    problems = [c for c in checks if c.status == "fail"]
    if problems:
        LOGGER.warning("验证发现 %d 项不通过", len(problems))
    return checks, problems
