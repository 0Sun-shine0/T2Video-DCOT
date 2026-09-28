"""本地账户与访问口令。

软件在本地保存接口密钥、并会把生成的分镜图像与视频片段落在工作目录里，
同一台机器可能由多人共用，因此启动时设一道本地口令闸门。

口令不保存明文：随机加盐后用 PBKDF2-HMAC-SHA256 迭代派生出摘要，
校验时重新派生再比对。这是真实的校验，不是形式上的输入框。
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

ACCOUNT_DIR = Path(__file__).resolve().parent.parent / "config"
ACCOUNT_FILE = ACCOUNT_DIR / "account.json"

# 迭代次数按桌面端登录的常见取值。提高它会线性增加单次校验耗时，
# 校验发生在启动时，用户能接受的等待上限决定了这个数不宜过大。
ITERATIONS = 200_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 6


class AccountError(RuntimeError):
    """账户读写或校验失败。"""


def _derive(password, salt, iterations):
    """派生口令摘要。同样的口令配同样的盐必然得到同样的结果。"""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _account_path(path=None):
    """取账户文件路径。参数可注入，便于测试用独立文件而不碰正式账户。"""
    return Path(path) if path else ACCOUNT_FILE


def exists(path=None):
    """是否已设置过账户。未设置时启动流程走"首次设置口令"。"""
    return _account_path(path).is_file()


def load(path=None):
    """读取账户记录。文件缺失或损坏时返回 None，由调用方决定如何引导。"""
    target = _account_path(path)
    if not target.is_file():
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.error("账户文件无法读取 %s: %s", target, exc)
        return None

    required = {"username", "salt", "digest", "iterations"}
    if not required <= set(record):
        LOGGER.error("账户文件字段不完整: %s", target)
        return None
    return record


def create(username, password, path=None, iterations=ITERATIONS):
    """建立账户。已存在时覆盖，供重置口令使用。"""
    username = (username or "").strip()
    if not username:
        raise AccountError("用户名不能为空")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(f"口令至少 {MIN_PASSWORD_LENGTH} 位")

    salt = secrets.token_bytes(SALT_BYTES)
    record = {
        "username": username,
        "salt": salt.hex(),
        "digest": _derive(password, salt, iterations).hex(),
        "iterations": iterations,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    target = _account_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        # 仅本人可读写。Windows 上 chmod 只影响只读位，这里尽力而为。
        os.chmod(target, 0o600)
    except OSError as exc:
        raise AccountError(f"账户文件写入失败: {exc}") from exc

    LOGGER.info("已建立本地账户: %s", username)
    return record


def verify(username, password, path=None):
    """校验用户名与口令。用 compare_digest 做常量时间比对，避免时序侧信道。"""
    record = load(path)
    if record is None:
        raise AccountError("尚未设置账户")

    if (username or "").strip() != record["username"]:
        return False

    salt = bytes.fromhex(record["salt"])
    expected = bytes.fromhex(record["digest"])
    actual = _derive(password or "", salt, int(record["iterations"]))
    return hmac.compare_digest(expected, actual)


def reset(path=None):
    """删除账户文件。忘记口令时的恢复途径，手册中有说明。"""
    target = _account_path(path)
    if target.is_file():
        target.unlink()
        LOGGER.info("已清除本地账户")
        return True
    return False
