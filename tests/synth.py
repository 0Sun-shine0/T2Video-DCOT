"""合成测试素材：确定性的图像与视频，不依赖任何外部资源。"""

from pathlib import Path

import cv2
import numpy as np

SIZE = (180, 320)  # (height, width)


def checkerboard(size=SIZE, block=20, low=30, high=225):
    """高对比度棋盘格：结构明显，拉普拉斯方差大。"""
    height, width = size
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = ((xs // block) + (ys // block)) % 2 == 0
    image = np.where(pattern, high, low).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


def solid(color=(128, 128, 128), size=SIZE):
    return np.full((*size, 3), color, dtype=np.uint8)


def gradient(size=SIZE):
    height, width = size
    ramp = np.linspace(20, 235, width, dtype=np.uint8)
    return np.tile(ramp, (height, 1))[:, :, None].repeat(3, axis=2)


def ball_frames(count=12, size=SIZE, background=40):
    """小球横向移动，光流幅值明显。"""
    height, width = size
    frames = []
    for index in range(count):
        frame = solid((background, background, background), size)
        cv2.circle(frame, (20 + index * 20, height // 2), 15, (0, 0, 230), -1)
        frames.append(frame)
    return frames


def static_frames(count=12, size=SIZE, color=(90, 90, 90)):
    """整段完全静止。"""
    return [solid(color, size) for _ in range(count)]


def noisy_frames(count=12, size=SIZE, seed=0):
    """逐帧独立随机噪声：帧间跳变极大。"""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (*size, 3), dtype=np.uint8) for _ in range(count)]


def write_image(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 与生产代码同理：cv2.imwrite 在 Windows 上处理不了含中文的路径
    ok, buffer = cv2.imencode(path.suffix or ".png", array)
    assert ok, f"编码失败: {path}"
    buffer.tofile(str(path))
    return path


def write_video(path, frames, fps=24):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "VideoWriter 无法打开"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    assert path.exists() and path.stat().st_size > 0, f"视频为空: {path}"
    return path
