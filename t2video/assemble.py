"""素材落盘与成片合成。"""

import logging
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import requests
from moviepy import VideoFileClip, concatenate_videoclips

from . import settings

LOGGER = logging.getLogger(__name__)

CHUNK = 8192
TEMP_PREFIX = "t2video_"

_ILLEGAL = str.maketrans({c: "" for c in '\\/:*?"<>|'})
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def safe_name(text, limit=50):
    """诗句转目录名的清洗。中文保留可读性，不做哈希——输出目录要能一眼看懂。"""
    cleaned = str(text or "").translate(_ILLEGAL)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    cleaned = "_".join(cleaned.split())[:limit].strip("._")

    if cleaned.upper() in _RESERVED:
        cleaned = "_" + cleaned
    return cleaned or "untitled"


def fetch(url, target, timeout=None, session=None):
    """流式下载。写完后校验字节数：空响应体不能算成功。

    原实现只看有没有抛异常，服务端返回 200 + 空 body 时会留下 0 字节文件，
    后面 moviepy 读它才报错，排查起来绕远路。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    caller = session or requests

    try:
        response = caller.get(url, stream=True,
                              timeout=timeout or settings.DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        written = 0
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
    except (requests.exceptions.RequestException, OSError) as exc:
        LOGGER.error("下载失败 %s: %s", str(url)[:70], exc)
        target.unlink(missing_ok=True)
        return False

    if written == 0:
        LOGGER.error("下载内容为空 %s", str(url)[:70])
        target.unlink(missing_ok=True)
        return False
    return True


def merge(clip_urls, output_dir, output_name, on_progress=None):
    """依次下载片段再拼接。返回成片路径，失败返回 None。"""
    if not clip_urls:
        LOGGER.error("没有可拼接的片段")
        return None

    workdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    clips = []
    try:
        local = []
        for order, url in enumerate(clip_urls, 1):
            if on_progress:
                on_progress(order, len(clip_urls))
            path = workdir / f"clip_{order}.mp4"
            if fetch(url, path):
                local.append(path)

        if not local:
            LOGGER.error("所有片段都下载失败")
            return None

        for path in local:
            try:
                clips.append(VideoFileClip(str(path)))
            except OSError as exc:
                LOGGER.warning("片段无法解析，跳过 %s: %s", path.name, exc)
        if not clips:
            LOGGER.error("没有可用的片段")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{safe_name(output_name)}.mp4"

        final = concatenate_videoclips(clips, method="compose")
        try:
            final.write_videofile(str(output_path), codec="libx264", audio_codec="aac",
                                  fps=24, threads=4, logger=None)
        finally:
            final.close()

        LOGGER.info("成片已输出: %s", output_path)
        return str(output_path)
    except OSError as exc:
        LOGGER.exception("拼接失败: %s", exc)
        return None
    finally:
        for clip in clips:
            try:
                clip.close()
            except OSError:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def read_frame(video_path, where="first"):
    """取首帧或末帧。用于验证时比对画面。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        if where == "last":
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def sweep_temp(max_age_hours=24):
    """清理本程序自己留下的临时目录。

    只认 TEMP_PREFIX 前缀。原实现遍历整个系统临时目录按时间删文件，
    会连带删掉其他程序正在用的东西，这里不做那种事。
    """
    root = Path(tempfile.gettempdir())
    if not root.is_dir():
        return 0

    deadline = time.time() - max_age_hours * 3600
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(TEMP_PREFIX):
            continue
        try:
            if entry.stat().st_mtime >= deadline:
                continue
            shutil.rmtree(entry)
            removed += 1
        except OSError as exc:
            LOGGER.warning("临时目录清理失败 %s: %s", entry, exc)
    return removed
