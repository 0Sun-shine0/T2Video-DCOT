"""全局设置：密钥、路径、风格词表、验证阈值。"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent
CUSTOM_OUTPUT_DIR = PROJECT_DIR / "Custom"

# 火山引擎视觉服务
API_HOST = "visual.volcengineapi.com"
API_REGION = "cn-north-1"
API_ENDPOINT = f"https://{API_HOST}"
API_SERVICE = "cv"
API_VERSION = "2022-08-31"


def _env(*names):
    """按顺序取第一个非空的环境变量。

    密钥名兼容两种写法：早期脚本用 ACCESS_KEY / SECRET_KEY，
    火山引擎控制台示例里常用 VOLC_ACCESS_KEY / VOLC_SECRET_KEY。
    只认一种会让另一种命名的 .env 静默失效，程序表面正常但拿不到密钥。
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


ACCESS_KEY = _env("ACCESS_KEY", "VOLC_ACCESS_KEY")
SECRET_KEY = _env("SECRET_KEY", "VOLC_SECRET_KEY")
DEEPSEEK_KEY = _env("DEEPSEEK_API_KEY")

CREDENTIAL_HINT = (
    "请在 .env 中配置密钥：ACCESS_KEY（或 VOLC_ACCESS_KEY）、"
    "SECRET_KEY（或 VOLC_SECRET_KEY）、DEEPSEEK_API_KEY"
)

TEXT2IMAGE_REQ_KEY = "jimeng_high_aes_general_v21_L"
IMAGE2VIDEO_REQ_KEY = "jimeng_vgfm_i2v_l20"
VIDEO_ASPECT_RATIO = "16:9"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 网络参数。重试只在两处发生：签名请求层退避、并发超限重试，不叠加。
HTTP_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 90
REQUEST_RETRY = 3
REQUEST_BACKOFF = 2.0
CONCURRENCY_RETRY = 5
CONCURRENCY_BACKOFF = 10.0
REQUESTS_PER_MINUTE = 10

POLL_MAX_ATTEMPTS = 30
POLL_INTERVAL = 15.0
POLL_MAX_INTERVAL = 300.0
POLL_GROWTH = 1.5

# 提示词上限按即梦平台限制取值。截断后缀计入总长，避免超限被拒。
PROMPT_LIMIT = 150
TRUNCATION_MARK = "…[截断]"

STYLE_PROMPTS = {
    "王昌龄风": ("边塞风光", "唐代边塞、风沙与军旅场景，色调沉稳，画面开阔。"),
    "陶渊明风": ("田园水墨", "魏晋田园、村舍与溪流，色彩清淡，留白自然。"),
    "李白风": ("盛唐气象", "长安、月色与歌舞，色彩明快，构图有动势。"),
    "其他": ("自定义风格", "根据输入内容安排场景、人物和色彩，避免现代元素。"),
}
DEFAULT_STYLE = "李白风"

# 跨分镜要延续的画面要素。这些子句会被逐条拼进每一次生成请求，
# 是"链"能维持住的关键；删掉任意一条都会让相邻画面明显脱节。
CONTINUITY_RULES = (
    "延续上一镜的地形与建筑轮廓",
    "保持光源方向与色温一致",
    "主要人物服饰、发型、佩饰保持不变",
    "镜头运动方向与前镜衔接，避免跳轴",
)

# 各风格对应的物理约束。古典题材最常崩的是重力、衣料和水面，
# 所以按风格给出不同的强调重点。
PHYSICS_RULES = {
    "王昌龄风": "沙尘随风向水平流动，旌旗与披风受重力自然下垂，马匹四蹄着地",
    "陶渊明风": "水面波纹向外扩散，草木随风摆动，炊烟竖直上升后缓慢弥散",
    "李白风": "衣袂与酒旗随风扬起，月光投影方向一致，舞袖遵循惯性",
    "其他": "遵守常规重力与流体规律，人体关节不出现反折",
}

VERIFY_THRESHOLDS = {
    "ssim_min": 0.45,          # 首帧与原图的结构相似度下限
    "brightness_delta_max": 40.0,
    "detail_min": 1.0,         # 边缘能量最强分块的均值，留白画面也能通过
    "frozen_flow": 0.15,       # 光流小于此值视为几乎没有位移
    "frozen_ratio_max": 0.50,  # 冻结帧占比超过此值判为静态
    "flicker_max": 45.0,       # 相邻帧平均绝对差上限
    "clip_continuity_min": 0.30,
}
VERIFY_BRIGHTNESS_RANGE = (15.0, 240.0)

LOGGER = logging.getLogger("t2video")


def configure_logging(level=logging.INFO):
    """给命令行入口用。GUI 会额外挂一个把日志送进界面的 handler。"""
    LOGGER.setLevel(level)
    if not LOGGER.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        LOGGER.addHandler(stream)
    return LOGGER
