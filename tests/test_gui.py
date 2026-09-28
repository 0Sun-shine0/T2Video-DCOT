"""界面测试。

用真实 Tk 根窗口做冒烟，重点是保证所有 _set 引用的控件名真实存在——
这类拼写错误没有界面测试时只会在用户点击时暴露。
"""

import logging
import shutil
import tkinter as tk

import pytest

from t2video import gui, settings
from synth import ball_frames, checkerboard, write_image


def fake_fetch(videos, image=None):
    """按目标后缀伪造下载：图片写真图，视频按队列依次取。"""
    queue = list(videos)
    picture = ball_frames()[0] if image is None else image

    def fetch(url, target, **kwargs):
        if str(target).endswith(".jpg"):
            write_image(target, picture)
        else:
            shutil.copyfile(queue.pop(0) if queue else videos[-1], target)
        return True

    return fetch


class StubClient:
    """不联网的客户端替身，记录调用但不发请求。"""

    def __init__(self):
        self.calls = []
        self.image = None
        self.video = None

    def text_to_image(self, prompt, reference=None):
        self.calls.append(("image", prompt))
        return self.image

    def image_to_video(self, image_url, motion_prompt=None):
        self.calls.append(("video", image_url))
        return self.video

CONTROLS = [
    "analyze_btn", "generate_btn", "cancel_btn", "save_prompts_btn",
    "play_btn", "pause_btn", "stop_btn",
    "image_save_btn", "image_view_btn", "clip_save_btn", "clip_play_btn",
    "final_save_btn", "final_play_btn",
    "custom_save_btn", "sequence_btn", "custom_video_btn", "custom_gen_btn",
]


@pytest.fixture(scope="module")
def tk_root():
    """整个模块共用一个根窗口。

    反复创建销毁 Tk 解释器会让 ttk 初始化随机失败
    （实测每轮 0~2 个用例报 "couldn't read file ttk/fonts.tcl"），只建一次。
    """
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"当前环境没有可用显示: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root, monkeypatch):
    for child in tk_root.winfo_children():
        child.destroy()

    # messagebox 是模态的，测试里必须拦下来，否则会一直等人点确定
    dialogs = []
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(gui.messagebox, name,
                            lambda *a, _n=name, **k: dialogs.append((_n, a)))

    window = gui.T2VideoWindow(tk_root, client=StubClient())
    window.dialogs = dialogs
    try:
        yield window
    finally:
        window._shut = True
        logging.getLogger("t2video").removeHandler(window._bridge)
        for child in tk_root.winfo_children():
            child.destroy()


BOARD = {
    "author": "王之涣", "era": "唐", "theme": "登楼", "background": "边塞",
    "style": "李白风",
    "verses": [],
    "shots": [
        {"index": 1, "original": "白日依山尽", "imagery": ["落日", "山"],
         "subject": "诗人", "scene": "山谷", "physics": "重力", "motion": "推移",
         "carried": {}, "absorbed": ["白日依山尽"]},
        {"index": 2, "original": "黄河入海流", "imagery": ["大河", "海"],
         "subject": "诗人", "scene": "河口", "physics": "重力", "motion": "推进",
         "carried": {"subject": "诗人", "scene": "山谷", "imagery": []},
         "absorbed": ["黄河入海流"]},
    ],
}


class TestConstruction:
    def test_every_control_exists(self, app):
        missing = [name for name in CONTROLS if getattr(app, name, None) is None]
        assert missing == [], f"缺少控件: {missing}"

    def test_set_accepts_all(self, app):
        app._set(**{name: tk.NORMAL for name in CONTROLS})

    def test_set_rejects_typos(self, app):
        """拼错控件名必须报错，否则状态设置会静默失效。"""
        with pytest.raises(AttributeError, match="nonexistent"):
            app._set(nonexistent=tk.NORMAL)

    def test_default_style_valid(self, app):
        assert app.style_var.get() in settings.STYLE_PROMPTS


class TestInput:
    def test_placeholder_is_empty(self, app):
        assert app._poem_text() == ""

    def test_real_text_returned(self, app):
        app.poem.delete("1.0", tk.END)
        app.poem.insert(tk.END, "  白日依山尽  ")
        assert app._poem_text() == "白日依山尽"


class TestBoardRendering:
    def test_renders_all_shots(self, app):
        app._render_board(BOARD)
        assert len(app.prompts) == 2
        assert len(app.shot_tree.get_children()) == 2
        assert app.author_label.cget("text").endswith("王之涣")
        assert "2 镜" in app.shot_label.cget("text")

    def test_tree_lists_verse_and_imagery(self, app):
        app._render_board(BOARD)
        first = app.shot_tree.item(app.shot_tree.get_children()[0])["values"]
        assert first[0] == "第1镜"
        assert first[1] == "白日依山尽"
        assert "落日" in first[2]

    def test_first_shot_selected_and_loaded(self, app):
        app._render_board(BOARD)
        assert app._editing == 0
        assert "落日" in app.image_prompt_box.get("1.0", tk.END)

    def test_prompts_are_distinct(self, app):
        app._render_board(BOARD)
        assert app.prompts[0]["image"] != app.prompts[1]["image"]
        assert app.prompts[0]["video"] != app.prompts[1]["video"]

    def test_second_shot_prompt_carries_context(self, app):
        """第二镜的提示词必须体现对前一镜的承接，这是"链"的直接体现。"""
        app._render_board(BOARD)
        assert "延续前一镜" in app.prompts[1]["image"]
        assert "延续前一镜" not in app.prompts[0]["image"]

    def test_switching_shot_stashes_edits(self, app):
        app._render_board(BOARD)
        app.image_prompt_box.delete("1.0", tk.END)
        app.image_prompt_box.insert(tk.END, "改过的提示词")
        app.shot_tree.selection_set(app.shot_tree.get_children()[1])
        app._on_shot_pick()
        assert app.prompts[0]["image"] == "改过的提示词"
        assert app._editing == 1

    def test_save_prompts_reads_editors(self, app):
        app._render_board(BOARD)
        app.image_prompt_box.delete("1.0", tk.END)
        app.image_prompt_box.insert(tk.END, "最终提示词")
        app.save_prompts()
        assert app.prompts[0]["image"] == "最终提示词"

    def test_empty_board_disables_save(self, app):
        app._render_board(dict(BOARD, shots=[]))
        assert str(app.save_prompts_btn.cget("state")) == "disabled"
        assert app.shot_tree.get_children() == ()


class TestProgressAndLog:
    def test_progress_widgets(self, app):
        app._progress(42, "处理中")
        assert int(float(app.progress["value"])) == 42
        assert app.progress_text.cget("text") == "处理中"

    def test_log_appends_and_locks(self, app):
        app._write_log("hello")
        assert "hello" in app.log.get("1.0", tk.END)
        assert str(app.log.cget("state")) == "disabled"

    def test_bridge_carries_logger_messages(self, app):
        logging.getLogger("t2video").info("来自 logger 的消息")
        app._pump()
        assert "来自 logger 的消息" in app.log.get("1.0", tk.END)

    def test_bridge_survives_broken_sink(self):
        def boom(_):
            raise RuntimeError("sink 挂了")

        bridge = gui.LogBridge(boom)
        record = logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None)
        bridge.emit(record)  # 不允许抛出


class TestAssetCallbacks:
    def test_shot_done_lists_and_enables(self, app):
        app._on_shot_done(1)
        assert app.image_list.size() == 1 and app.clip_list.size() == 1
        assert str(app.image_save_btn.cget("state")) == "normal"
        assert str(app.clip_play_btn.cget("state")) == "normal"

    def test_finish_fills_entry(self, app):
        app._finish("/tmp/final.mp4")
        assert app.final_entry.get() == "/tmp/final.mp4"
        assert str(app.final_entry.cget("state")) == "readonly"
        assert str(app.play_btn.cget("state")) == "normal"


class TestCancel:
    def test_stop_sets_event(self, app):
        app.cancel.clear()
        app.stop()
        assert app.cancel.is_set()
        assert str(app.cancel_btn.cget("state")) == "disabled"
        assert str(app.generate_btn.cget("state")) == "normal"


class TestGenerateGuards:
    def test_requires_analysis(self, app):
        app.poem.delete("1.0", tk.END)
        app.poem.insert(tk.END, "白日依山尽")
        app.board = None
        app.generate()
        assert any(kind == "showinfo" for kind, _ in app.dialogs)

    def test_requires_poem(self, app):
        app.generate()
        assert any(kind == "showerror" for kind, _ in app.dialogs)

    def test_save_prompts_does_not_block(self, app):
        """save_prompts 会弹提示框；弹窗必须是拦下来的，否则测试会卡住等人点击。"""
        app._render_board(BOARD)
        app.save_prompts()
        assert any(kind == "showinfo" for kind, _ in app.dialogs)


class TestRenderPipeline:
    """用替身客户端走一遍真实生成路径，验证链式串联与自我修正。"""

    def prepare(self, app, tmp_path):
        app.board = dict(BOARD)
        app._render_board(app.board)
        app._images_dir = tmp_path / "分镜图像"
        app._clips_dir = tmp_path / "视频片段"
        app._images_dir.mkdir(parents=True, exist_ok=True)
        app._clips_dir.mkdir(parents=True, exist_ok=True)
        app._total = len(app.board["shots"])
        app.client.calls.clear()
        return app.client

    def test_missing_image_skips_shot(self, app, tmp_path):
        client = self.prepare(app, tmp_path)
        client.image = None
        assert app._render_shot(app.board["shots"][0], app.prompts[0], 0) is None

    def test_missing_video_skips_shot(self, app, tmp_path):
        client = self.prepare(app, tmp_path)
        client.image = "http://i"
        client.video = None
        assert app._render_shot(app.board["shots"][0], app.prompts[0], 0) is None

    def test_successful_shot_records_assets(self, app, tmp_path, monkeypatch, moving_video):
        client = self.prepare(app, tmp_path)
        client.image = "http://i"
        client.video = "http://v"
        monkeypatch.setattr(gui.assemble, "fetch", fake_fetch([moving_video]))
        result = app._render_shot(app.board["shots"][0], app.prompts[0], 0)
        app._pump()  # 界面更新是投递到主线程的，断言前要先排空队列
        assert result is not None
        assert result["video_url"] == "http://v"
        assert app.assets["images"] == ["http://i"]
        assert app.image_list.size() == 1
        assert str(app.clip_play_btn.cget("state")) == "normal"

    def test_failed_verification_triggers_repair(self, app, tmp_path, monkeypatch,
                                                 moving_video, static_video):
        """验证不通过时应当带着修正要求重生成，且以更严格的提示词重试。"""
        client = self.prepare(app, tmp_path)
        client.image = "http://i"
        client.video = "http://v"

        # 第一次给静态视频（运动验证必挂），第二次给运动视频
        monkeypatch.setattr(gui.assemble, "fetch",
                            fake_fetch([static_video, moving_video]))

        app._render_shot(app.board["shots"][0], app.prompts[0], 0)

        image_calls = [c for c in client.calls if c[0] == "image"]
        assert len(image_calls) >= 2, "验证失败后应当带修正重新生成"
        assert image_calls[0][1] != image_calls[1][1], "重生成时提示词应当已被修正"

    def test_repair_rounds_bounded(self, app, tmp_path, monkeypatch, static_video):
        """修正只做有限轮，否则同一镜会反复烧额度。"""
        client = self.prepare(app, tmp_path)
        client.image = "http://i"
        client.video = "http://v"
        monkeypatch.setattr(gui.assemble, "fetch",
                            fake_fetch([static_video] * 6))
        app._render_shot(app.board["shots"][0], app.prompts[0], 0)
        video_calls = [c for c in client.calls if c[0] == "video"]
        assert len(video_calls) <= 1 + gui.REPAIR_ROUNDS
