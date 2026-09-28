"""图形界面，并把整条动态思维链串起来。

生成流程不是"一条道走到黑"：每一镜出图、出片之后立刻验证，不通过的镜头
会把修正要求并回自己的提示词再重生成一次。修正只做一轮，避免同一镜反复烧额度。
"""

import ctypes
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from . import account, assemble, chain, prompt_builder, settings, verify, visual_api

LOGGER = logging.getLogger(__name__)

SOFTWARE_TITLE = "基于动态思维链的文生视频方法研究系统"
SOFTWARE_VERSION = "V1.0"
SOFTWARE_HEADER = f"{SOFTWARE_TITLE} {SOFTWARE_VERSION}"

PLACEHOLDER = "输入要生成视频的诗词..."
FRAME_INTERVAL = 50
PREVIEW_SIZE = (800, 450)

# 单镜允许的自我修正轮数。置 0 即关闭反馈，退化成单向流水线。
REPAIR_ROUNDS = 1


class LoginWindow:
    """启动时的口令闸门：没有账户就引导设置，已有账户就校验。

    做成独立窗口而不是主窗口上的遮罩，是为了让登录前后的界面职责清楚：
    校验通过才构造主窗口，未通过时主窗口根本不创建，
    也就不存在"绕过登录直接操作界面"的路径。
    """

    def __init__(self, root):
        self.root = root
        self.passed = False
        self._first_run = not account.exists()

        root.title(f"{SOFTWARE_HEADER} 　登录")
        root.geometry("560x420")
        root.resizable(False, False)

        self._build()

    def _build(self):
        """搭登录界面：顶部软件名称横幅，中部账号口令表单，底部按钮与说明。"""
        banner = tk.Frame(self.root, bg="#1F4E79", height=88)
        banner.pack(fill=tk.X)
        banner.pack_propagate(False)
        tk.Label(banner, text=SOFTWARE_TITLE, bg="#1F4E79", fg="white",
                 font=("黑体", 15, "bold")).pack(pady=(18, 2))
        tk.Label(banner, text=f"版本 {SOFTWARE_VERSION}", bg="#1F4E79", fg="#C7D9EC",
                 font=("黑体", 10)).pack()

        body = ttk.Frame(self.root, padding=20)
        body.pack(fill=tk.BOTH, expand=True)

        title = "首次使用，请设置访问口令" if self._first_run else "请输入访问口令"
        ttk.Label(body, text=title, font=("黑体", 12)).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 14))

        ttk.Label(body, text="用户名").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.user_entry = ttk.Entry(body, width=26)
        self.user_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)
        self.user_entry.insert(0, "user" if self._first_run else "")

        ttk.Label(body, text="口令").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.password_entry = ttk.Entry(body, width=26, show="●")
        self.password_entry.grid(row=2, column=1, sticky=tk.EW, pady=4)

        self.confirm_entry = None
        if self._first_run:
            ttk.Label(body, text="确认口令").grid(row=3, column=0, sticky=tk.W, pady=4)
            self.confirm_entry = ttk.Entry(body, width=26, show="●")
            self.confirm_entry.grid(row=3, column=1, sticky=tk.EW, pady=4)

        body.columnconfigure(1, weight=1)

        self.hint = ttk.Label(body, text="", foreground="#c0504d", wraplength=430)
        self.hint.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky=tk.E, pady=(16, 0))

        self.submit_btn = ttk.Button(
            buttons, text="设置并进入" if self._first_run else "登 录",
            width=14, command=self.submit)
        self.submit_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="退出", width=10,
                   command=self._cancel).pack(side=tk.LEFT, padx=4)

        note = ("口令用于保护本机保存的接口密钥与生成素材，至少 "
                f"{account.MIN_PASSWORD_LENGTH} 位。"
                if self._first_run else
                "忘记口令可删除 config/account.json 后重新设置。")
        ttk.Label(body, text=note, foreground="#666666", wraplength=430).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        self.root.bind("<Return>", lambda _event: self.submit())
        self.password_entry.focus_set()

    def _cancel(self):
        """用户主动退出。不建立账户也不创建主窗口。"""
        self.passed = False
        self.root.destroy()

    def submit(self):
        """首次运行走建立账户，之后走校验；两条路径都以真实密码学操作为准。"""
        username = self.user_entry.get().strip()
        password = self.password_entry.get()

        try:
            if self._first_run:
                confirm = self.confirm_entry.get() if self.confirm_entry else ""
                if password != confirm:
                    self.hint.config(text="两次输入的口令不一致")
                    return
                account.create(username, password)
            else:
                if not account.verify(username, password):
                    LOGGER.warning("口令校验未通过，用户名: %s", username or "(空)")
                    self.hint.config(text="用户名或口令不正确")
                    self.password_entry.delete(0, tk.END)
                    return
        except account.AccountError as exc:
            self.hint.config(text=str(exc))
            return

        LOGGER.info("登录成功: %s", username)
        self.passed = True
        self.root.destroy()


def ask_login(root):
    """弹出登录窗口，返回通过校验的用户名，未通过返回 None。

    返回用户名而不是布尔值：主窗口页眉要显示当前登录者，
    返回值取自校验成功的那条记录，不依赖界面控件的残留状态。
    """
    window = LoginWindow(root)
    root.mainloop()
    if not window.passed:
        return None
    return account.load()["username"]


class LogBridge(logging.Handler):
    """把日志送进界面。后端和界面共用一条日志链路，不用两套输出。"""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        """转发一条日志。界面已销毁等情况下 sink 会抛错，
        交给 handleError 处理，绝不让日志本身把程序带崩。
        """
        try:
            self.sink(self.format(record))
        except Exception:
            self.handleError(record)


class T2VideoWindow:
    """主窗口，同时承担整条动态思维链的调度。

    界面状态分三类存放：assets 记录本次生成产物，prompts 与分镜一一对应
    且允许用户改写，style_previews 与 custom_images 是两处独立的小图缓存。
    生成任务的临时状态（目录、总数）挂在实例上，避免每个渲染函数都拖一串参数。
    """

    def __init__(self, root, client=None, username=""):
        self.root = root
        self.queue = queue.Queue()
        self.client = client or visual_api.VisualApiClient()
        self.username = username

        self.cancel = threading.Event()
        self.busy = False
        self._shut = False
        self._cap = None
        self._playing = False
        self._frame_lock = threading.Lock()
        self._frame = None

        self.style_previews = {}
        self.board = None
        self.prompts = []          # 与 board["shots"] 一一对应，用户可编辑
        self._editing = None       # 当前正在编辑的分镜下标
        self.custom_images = []
        self.assets = {"images": [], "clips": [], "final": None}

        # 一次生成任务的临时状态。放在实例上而不是层层传参，
        # 免得每个渲染函数都拖着七八个参数。
        self._images_dir = settings.PROJECT_DIR
        self._clips_dir = settings.PROJECT_DIR
        self._total = 0

        self._build()
        self._attach_log()
        self.root.after(FRAME_INTERVAL, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    # 主线程调度

    def post(self, fn):
        """把回调排进主线程队列。工作线程只能通过它与界面交互。"""
        self.queue.put(fn)

    def _pump(self):
        """定时排空回调队列并渲染最新一帧。只允许主线程调用。"""
        if self._shut:
            return
        for _ in range(200):
            try:
                fn = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except tk.TclError as exc:
                LOGGER.debug("界面更新丢弃: %s", exc)
        self._draw_frame()
        self.root.after(FRAME_INTERVAL, self._pump)

    def _attach_log(self):
        """把后台日志接到界面文本框。单次上限 200 条，避免一次涌入卡住界面。"""
        settings.configure_logging()
        self._bridge = LogBridge(lambda text: self.post(lambda: self._write_log(text)))
        self._bridge.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self._bridge.setLevel(logging.INFO)
        logging.getLogger("t2video").addHandler(self._bridge)

    def _write_log(self, text):
        """追加一行带时间戳的日志，随后恢复只读避免用户误改。"""
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _draw_frame(self):
        """取走单槽里的最新一帧并显示。无线程时直接返回。"""
        with self._frame_lock:
            frame, self._frame = self._frame, None
        if frame is None:
            return
        photo = ImageTk.PhotoImage(frame)
        self.screen.config(image=photo)
        self.screen.image = photo

    def _set(self, **states):
        """批量切换控件状态。控件名拼错时立刻报错，否则设置会静默失效。"""
        for name, state in states.items():
            widget = getattr(self, name, None)
            if widget is None:
                raise AttributeError(f"没有名为 {name!r} 的控件")
            widget.config(state=state)

    def _close(self):
        """关窗清理：停掉播放与生成，摘掉日志 handler，再销毁窗口。"""
        self._shut = True
        self.cancel.set()
        self._playing = False
        if self._cap:
            self._cap.release()
        logging.getLogger("t2video").removeHandler(self._bridge)
        self.root.destroy()

    # 界面搭建

    def _build(self):
        """搭建窗口骨架：顶部软件名称页眉，下方左右两个面板。

        顶部页眉常驻软件全称与版本号：说明书里的界面截图必须与登记的
        软件名称一致，把名称画在窗口内部，截图就必然带上它，
        不依赖操作系统的标题栏。
        """
        self.root.title(SOFTWARE_HEADER)
        self.root.geometry("1360x920")

        banner = tk.Frame(self.root, bg="#1F4E79", height=52)
        banner.pack(fill=tk.X)
        banner.pack_propagate(False)
        tk.Label(banner, text=SOFTWARE_HEADER, bg="#1F4E79", fg="white",
                 font=("黑体", 14, "bold")).pack(side=tk.LEFT, padx=16)
        self.user_label = tk.Label(banner, text="", bg="#1F4E79", fg="#C7D9EC",
                                   font=("黑体", 10))
        self.user_label.pack(side=tk.RIGHT, padx=16)

        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        left = ttk.LabelFrame(outer, text="输入与控制", padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=4)
        right = ttk.LabelFrame(outer, text="输出与预览", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        self._build_left(left)
        self._build_right(right)
        self.user_label.config(text=f"当前用户：{self.username}" if self.username else "")

    def _build_left(self, parent):
        """左侧面板：风格选择与预览、诗词输入、操作按钮、进度条、处理日志。

        多处文本框显式指定 width=32。tk.Text 默认 80 字符宽，
        不指定会把左面板撑开并挤掉右侧的表格列。
        """
        style_box = ttk.LabelFrame(parent, text="艺术风格", padding=8)
        style_box.pack(fill=tk.X, padx=4, pady=4)

        row = ttk.Frame(style_box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="风格:").pack(side=tk.LEFT, padx=(0, 8))
        self.style_var = tk.StringVar(value=settings.DEFAULT_STYLE)
        self.style_box = ttk.Combobox(row, textvariable=self.style_var,
                                      values=list(settings.STYLE_PROMPTS), state="readonly", width=14)
        self.style_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_box.bind("<<ComboboxSelected>>", self.on_style)

        self.style_image = ttk.Label(style_box, text="选择风格后显示预览",
                                     anchor=tk.CENTER, relief=tk.SUNKEN, padding=8)
        self.style_image.pack(fill=tk.BOTH, expand=True, pady=6)

        self.style_note = scrolledtext.ScrolledText(style_box, height=4, width=32, wrap=tk.WORD,
                                                    state=tk.DISABLED)
        self.style_note.pack(fill=tk.BOTH)

        poem_box = ttk.LabelFrame(parent, text="诗词输入", padding=8)
        poem_box.pack(fill=tk.BOTH, padx=4, pady=4)
        self.poem = scrolledtext.ScrolledText(poem_box, height=8, width=32, wrap=tk.WORD)
        self.poem.pack(fill=tk.BOTH, expand=True)
        self.poem.insert(tk.END, PLACEHOLDER)

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, padx=4, pady=4)
        self.analyze_btn = ttk.Button(buttons, text="解析诗词", width=12, command=self.analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=3)
        self.generate_btn = ttk.Button(buttons, text="生成视频", width=12, command=self.generate)
        self.generate_btn.pack(side=tk.LEFT, padx=3)
        self.cancel_btn = ttk.Button(buttons, text="取消", width=12, state=tk.DISABLED,
                                     command=self.stop)
        self.cancel_btn.pack(side=tk.LEFT, padx=3)

        progress_box = ttk.LabelFrame(parent, text="处理进度", padding=8)
        progress_box.pack(fill=tk.X, padx=4, pady=4)
        self.progress_text = ttk.Label(progress_box, text="准备就绪")
        self.progress_text.pack(fill=tk.X)
        self.progress = ttk.Progressbar(progress_box, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, pady=4)

        log_box = ttk.LabelFrame(parent, text="处理日志", padding=8)
        log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log = scrolledtext.ScrolledText(log_box, height=12, width=32)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.config(state=tk.DISABLED)

    def _build_right(self, parent):
        """右侧面板：四个页签，依次为分镜与提示词、视频预览、生成资源、自定义生成。"""
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.tab_shots = ttk.Frame(self.tabs)
        self.tab_preview = ttk.Frame(self.tabs)
        self.tab_assets = ttk.Frame(self.tabs)
        self.tab_custom = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_shots, text="分镜与提示词")
        self.tabs.add(self.tab_preview, text="视频预览")
        self.tabs.add(self.tab_assets, text="生成资源")
        self.tabs.add(self.tab_custom, text="自定义生成")

        self._build_shots_tab()
        self._build_preview_tab()
        self._build_assets_tab()
        self._build_custom_tab()

    def _build_shots_tab(self):
        """分镜页签：上方作者年代与主题，中部 Treeview 总览，下方提示词编辑框。

        总览用 Treeview 而不是自绘 Frame 网格：列宽固定且自带滚动，
        分镜多时不会把面板撑破，选中事件也比逐行绑定更省事。
        """
        box = ttk.LabelFrame(self.tab_shots, text="意象抽取与分镜规划结果", padding=8)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        head = ttk.Frame(box)
        head.pack(fill=tk.X)
        self.author_label = ttk.Label(head, text="作者：—")
        self.author_label.pack(side=tk.LEFT, padx=6)
        self.era_label = ttk.Label(head, text="年代：—")
        self.era_label.pack(side=tk.LEFT, padx=6)
        self.shot_label = ttk.Label(head, text="分镜：—")
        self.shot_label.pack(side=tk.LEFT, padx=6)

        self.theme_text = scrolledtext.ScrolledText(box, height=3, width=40, wrap=tk.WORD,
                                                    state=tk.DISABLED)
        self.theme_text.pack(fill=tk.X, pady=6)

        # 分镜总览用 Treeview：列宽固定、自带滚动，行数多也不会撑破布局
        overview = ttk.LabelFrame(box, text="分镜列表（单击某行查看并编辑该镜提示词）", padding=6)
        overview.pack(fill=tk.BOTH, expand=True, pady=4)
        self.shot_tree = ttk.Treeview(overview, columns=("shot", "verse", "imagery"),
                                      show="headings", height=6)
        for key, title, width, grow in (("shot", "镜号", 52, False),
                                        ("verse", "原句", 140, False),
                                        ("imagery", "核心意象", 230, True)):
            self.shot_tree.heading(key, text=title)
            self.shot_tree.column(key, width=width, anchor=tk.W, stretch=grow)
        tree_bar = ttk.Scrollbar(overview, orient=tk.VERTICAL, command=self.shot_tree.yview)
        self.shot_tree.configure(yscrollcommand=tree_bar.set)
        self.shot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.shot_tree.bind("<<TreeviewSelect>>", self._on_shot_pick)

        editor = ttk.LabelFrame(box, text="提示词编辑", padding=6)
        editor.pack(fill=tk.BOTH, expand=True, pady=4)
        editor.columnconfigure(0, weight=1)
        self.image_prompt_box = self._prompt_field(editor, "图像提示词", 0)
        self.video_prompt_box = self._prompt_field(editor, "视频提示词", 1)

        foot = ttk.Frame(box)
        foot.pack(fill=tk.X)
        self.save_prompts_btn = ttk.Button(foot, text="保存提示词修改", state=tk.DISABLED,
                                           command=self.save_prompts)
        self.save_prompts_btn.pack(side=tk.RIGHT, padx=4)

    def _prompt_field(self, parent, title, row):
        """造一行「标签 + 多行编辑框」。两行共用同一列的权重，随窗口一起伸缩。"""
        line = ttk.Frame(parent)
        line.grid(row=row, column=0, sticky=tk.EW, pady=2)
        ttk.Label(line, text=title, width=11, anchor=tk.W).pack(side=tk.LEFT)
        # 显式给宽度：tk.Text 默认 80 字符，会把右面板撑到窗口之外
        widget = tk.Text(line, height=3, width=40, wrap=tk.WORD)
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return widget

    def _on_shot_pick(self, event=None):
        """切换选中分镜：先把编辑框里未保存的内容收回，再载入新的一镜。"""
        picked = self.shot_tree.selection()
        if not picked:
            return
        index = self.shot_tree.index(picked[0])
        if index == self._editing:
            return

        self._stash_editors()
        self._editing = index
        entry = self.prompts[index]
        for kind, box in (("image", self.image_prompt_box), ("video", self.video_prompt_box)):
            box.delete("1.0", tk.END)
            box.insert(tk.END, entry[kind])

    def _stash_editors(self):
        """把编辑框里的内容收回 prompts。

        切换分镜、保存、开始生成之前都要调用一次，
        否则用户改了提示词却没点保存时，改动会在切换分镜的瞬间丢失。
        """
        if self._editing is None or self._editing >= len(self.prompts):
            return
        entry = self.prompts[self._editing]
        entry["image"] = self.image_prompt_box.get("1.0", tk.END).strip()
        entry["video"] = self.video_prompt_box.get("1.0", tk.END).strip()

    def _build_preview_tab(self):
        """预览页签：一个画面显示区与播放、暂停、停止三个按钮。"""
        box = ttk.LabelFrame(self.tab_preview, text="成片预览", padding=8)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        holder = ttk.Frame(box)
        holder.pack(fill=tk.BOTH, expand=True)
        self.screen = ttk.Label(holder, text="生成完成后在此播放", anchor=tk.CENTER,
                                relief=tk.SUNKEN, padding=8)
        self.screen.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(box)
        controls.pack(fill=tk.X, pady=4)
        self.play_btn = ttk.Button(controls, text="播放", state=tk.DISABLED, command=self.play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        self.pause_btn = ttk.Button(controls, text="暂停", state=tk.DISABLED, command=self.pause)
        self.pause_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(controls, text="停止", state=tk.DISABLED, command=self.halt)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

    def _build_assets_tab(self):
        """资源页签：分镜图像列表、视频片段列表、成片路径，各带下载与播放按钮。"""
        box = ttk.LabelFrame(self.tab_assets, text="生成资源", padding=8)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        img_box = ttk.LabelFrame(box, text="分镜图像", padding=8)
        img_box.pack(fill=tk.BOTH, pady=4)
        row = ttk.Frame(img_box)
        row.pack(fill=tk.BOTH)
        self.image_list = tk.Listbox(row, height=7, selectmode=tk.SINGLE)
        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar = ttk.Scrollbar(row, orient=tk.VERTICAL, command=self.image_list.yview)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list.config(yscrollcommand=bar.set)
        btns = ttk.Frame(img_box)
        btns.pack(fill=tk.X, pady=4)
        self.image_save_btn = ttk.Button(btns, text="下载选中图像", state=tk.DISABLED,
                                         command=self.save_image)
        self.image_save_btn.pack(side=tk.LEFT, padx=4)
        self.image_view_btn = ttk.Button(btns, text="预览图像", state=tk.DISABLED,
                                         command=self.view_image)
        self.image_view_btn.pack(side=tk.LEFT, padx=4)

        clip_box = ttk.LabelFrame(box, text="视频片段", padding=8)
        clip_box.pack(fill=tk.BOTH, pady=4)
        row2 = ttk.Frame(clip_box)
        row2.pack(fill=tk.BOTH)
        self.clip_list = tk.Listbox(row2, height=7, selectmode=tk.SINGLE)
        self.clip_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar2 = ttk.Scrollbar(row2, orient=tk.VERTICAL, command=self.clip_list.yview)
        bar2.pack(side=tk.RIGHT, fill=tk.Y)
        self.clip_list.config(yscrollcommand=bar2.set)
        btns2 = ttk.Frame(clip_box)
        btns2.pack(fill=tk.X, pady=4)
        self.clip_save_btn = ttk.Button(btns2, text="下载片段", state=tk.DISABLED,
                                        command=self.save_clip)
        self.clip_save_btn.pack(side=tk.LEFT, padx=4)
        self.clip_play_btn = ttk.Button(btns2, text="播放片段", state=tk.DISABLED,
                                        command=self.play_clip)
        self.clip_play_btn.pack(side=tk.LEFT, padx=4)

        final_box = ttk.LabelFrame(box, text="成片", padding=8)
        final_box.pack(fill=tk.X, pady=4)
        self.final_entry = ttk.Entry(final_box)
        self.final_entry.pack(fill=tk.X, pady=4)
        self.final_entry.config(state="readonly")
        btns3 = ttk.Frame(final_box)
        btns3.pack(fill=tk.X)
        self.final_save_btn = ttk.Button(btns3, text="下载成片", state=tk.DISABLED,
                                         command=self.save_final)
        self.final_save_btn.pack(side=tk.LEFT, padx=4)
        self.final_play_btn = ttk.Button(btns3, text="播放成片", state=tk.DISABLED,
                                         command=self.play_final)
        self.final_play_btn.pack(side=tk.LEFT, padx=4)

    def _build_custom_tab(self):
        """自定义页签：关键词输入、预览图、已生成图列表与三种生成按钮。

        该页签不经过意象抽取与分镜规划，关键词直接作为画面描述，
        用于快速试画面，或生成与诗词无关的素材。
        """
        box = ttk.LabelFrame(self.tab_custom, text="自定义关键词生成", padding=8)
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="关键词:").pack(side=tk.LEFT)
        self.keyword = tk.StringVar()
        ttk.Entry(row, textvariable=self.keyword, width=36).pack(side=tk.LEFT, padx=6, fill=tk.X,
                                                                expand=True)
        self.custom_gen_btn = ttk.Button(row, text="生成预览图", command=self.custom_preview)
        self.custom_gen_btn.pack(side=tk.LEFT, padx=4)

        self.custom_status = ttk.Label(box, text="等待输入")
        self.custom_status.pack(fill=tk.X, pady=4)

        holder = ttk.Frame(box)
        holder.pack(fill=tk.BOTH, expand=True)
        self.custom_image = ttk.Label(holder, text="生成后在此显示", anchor=tk.CENTER,
                                      relief=tk.SUNKEN, padding=8)
        self.custom_image.pack(fill=tk.BOTH, expand=True)

        self.custom_list = tk.Listbox(box, height=4, selectmode=tk.SINGLE)
        self.custom_list.pack(fill=tk.X, pady=4)

        btns = ttk.Frame(box)
        btns.pack(fill=tk.X)
        self.custom_save_btn = ttk.Button(btns, text="下载选中图像", state=tk.DISABLED,
                                          command=self.custom_save)
        self.custom_save_btn.pack(side=tk.LEFT, padx=4)
        self.sequence_btn = ttk.Button(btns, text="生成序列片段", state=tk.DISABLED,
                                       command=self.custom_sequence)
        self.sequence_btn.pack(side=tk.LEFT, padx=4)
        self.custom_video_btn = ttk.Button(btns, text="生成完整视频", state=tk.DISABLED,
                                           command=self.custom_video)
        self.custom_video_btn.pack(side=tk.LEFT, padx=4)

    # 风格

    def on_style(self, event=None):
        """切换风格：刷新说明文字，并异步补一张预览图。

        预览图有副作用（要调接口），所以命中缓存就不重复生成；
        "其他"是自定义风格，没有固定画面描述，不生成预览。
        """
        name = self.style_var.get()
        desc, detail = settings.STYLE_PROMPTS.get(name, ("", ""))
        physics = settings.PHYSICS_RULES.get(name, "")
        self.style_note.config(state=tk.NORMAL)
        self.style_note.delete("1.0", tk.END)
        self.style_note.insert(tk.END, f"{desc}\n{detail}\n物理约束：{physics}")
        self.style_note.config(state=tk.DISABLED)

        if name in self.style_previews:
            self._show_style_preview(name)
        elif name != "其他":
            self._thread(self._make_style_preview, name)

    def _make_style_preview(self, name):
        """工作线程：取图并落盘到 Custom/风格预览，再把 PIL 图交回主线程。

        不在本函数里创建 PhotoImage——Tk 对象只能在主线程建，
        在工作线程里建可能直接让程序崩掉。
        """
        url = self.client.text_to_image(prompt_builder.preview_prompt(name))
        if not url:
            self.post(lambda: self.style_image.config(image="", text="预览图生成失败"))
            return
        target = settings.CUSTOM_OUTPUT_DIR / "风格预览" / f"{assemble.safe_name(name)}.jpg"
        if not assemble.fetch(url, target):
            self.post(lambda: self.style_image.config(image="", text="预览图下载失败"))
            return
        try:
            image = Image.open(target)
            image.load()
        except OSError as exc:
            LOGGER.error("预览图无法打开: %s", exc)
            return
        image = ImageOps.contain(image, (400, 200))
        self.post(lambda: self._store_style_preview(name, image))

    def _store_style_preview(self, name, image):
        """主线程：创建 PhotoImage 并留一份引用。

        必须保留引用，否则 Python 回收该对象后界面上的图会变空白。
        """
        photo = ImageTk.PhotoImage(image)
        self.style_previews[name] = photo
        self._show_style_preview(name)

    def _show_style_preview(self, name):
        """把缓存里的预览图贴到标签上。没有缓存则保持原有提示文字。"""
        photo = self.style_previews.get(name)
        if photo:
            self.style_image.config(image=photo, text="")
            self.style_image.image = photo

    # 解析

    def _poem_text(self):
        """取输入框内容。仍是占位提示文字时按空串处理，避免把提示语当诗解析。"""
        text = self.poem.get("1.0", tk.END).strip()
        return "" if text == PLACEHOLDER else text

    def analyze(self):
        """校验输入并在工作线程里启动解析。主线程只做校验与按钮状态。"""
        poem = self._poem_text()
        if not poem:
            messagebox.showerror("错误", "请先输入诗词")
            return
        self._set(analyze_btn=tk.DISABLED, generate_btn=tk.DISABLED)
        self._thread(self._do_analyze, poem)

    def _do_analyze(self, poem):
        """工作线程：调解析接口，成功后把分镜表交回主线程渲染。"""
        try:
            board = chain.build_storyboard(poem, self.style_var.get())
        except chain.ChainError as exc:
            LOGGER.error("诗词解析失败: %s", exc)
            if exc.raw:
                LOGGER.error("原始响应: %s", exc.raw)
            self.post(lambda: messagebox.showerror("解析失败", str(exc)))
            return
        except Exception as exc:
            LOGGER.exception("诗词解析异常")
            self.post(lambda: messagebox.showerror("解析异常", str(exc)))
            return
        finally:
            self.post(lambda: self._set(analyze_btn=tk.NORMAL, generate_btn=tk.NORMAL))

        self.board = board
        LOGGER.info("意象抽取完成：作者=%s，年代=%s，共 %d 句 / %d 镜",
                    board["author"] or "未知", board["era"] or "未知",
                    len(board["verses"]), len(board["shots"]))
        self.post(lambda: self._render_board(board))

    def _render_board(self, board):
        """把分镜表画到界面上，并为每一镜预置图像与视频提示词。

        prompts 与 board["shots"] 严格同序同长，后续生成按下标取用，
        因此这里每次都要重建，不能增量追加。
        """
        self.tabs.select(self.tab_shots)
        self.author_label.config(text=f"作者：{board['author'] or '未知'}")
        self.era_label.config(text=f"年代：{board['era'] or '未知'}")
        self.shot_label.config(text=f"分镜：{len(board['shots'])} 镜")

        self.theme_text.config(state=tk.NORMAL)
        self.theme_text.delete("1.0", tk.END)
        self.theme_text.insert(tk.END, f"主题：{board['theme']}\n背景：{board['background']}")
        self.theme_text.config(state=tk.DISABLED)

        self.shot_tree.delete(*self.shot_tree.get_children())
        self.prompts = []
        self._editing = None

        for shot in board["shots"]:
            self.prompts.append({
                "image": prompt_builder.image_prompt(shot, board),
                "video": prompt_builder.video_prompt(shot, board),
            })
            self.shot_tree.insert("", tk.END, values=(
                f"第{shot['index']}镜",
                shot.get("absorbed", [shot["original"]])[0],
                "、".join(shot["imagery"]),
            ))

        self.save_prompts_btn.config(state=tk.NORMAL if board["shots"] else tk.DISABLED)

        children = self.shot_tree.get_children()
        if children:
            self.shot_tree.selection_set(children[0])
            self.shot_tree.focus(children[0])
            self._on_shot_pick()

    def save_prompts(self):
        """把编辑框内容写回 prompts 并提示用户。"""
        self._stash_editors()
        LOGGER.info("已保存 %d 镜的提示词修改", len(self.prompts))
        messagebox.showinfo("已保存", "提示词修改已保存")

    # 生成

    def generate(self):
        """校验前置条件并在工作线程里启动整条生成链。

        界面上必须已经解析出分镜，否则没有提示词可用；
        生成期间禁用生成与解析按钮，只留取消可用。
        """
        if self.busy:
            messagebox.showwarning("提示", "已有任务在进行")
            return
        poem = self._poem_text()
        if not poem:
            messagebox.showerror("错误", "请先输入诗词")
            return
        if not self.board or not self.board.get("shots"):
            messagebox.showinfo("提示", "请先点击「解析诗词」")
            return

        self.cancel.clear()
        self.busy = True
        self._set(generate_btn=tk.DISABLED, cancel_btn=tk.NORMAL, analyze_btn=tk.DISABLED)
        self._thread(self._run, poem)

    def _run(self, poem):
        """工作线程：逐镜生成再拼接，是整条链的驱动。

        单镜失败只跳过该镜，不中断整条链；全部失败才判定本次生成失败。
        无论成功失败都会在 finally 里把按钮状态复位，避免界面卡在"处理中"。
        """
        board = self.board
        project = settings.PROJECT_DIR / assemble.safe_name(poem)
        self._images_dir = project / "分镜图像"
        self._clips_dir = project / "视频片段"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._total = len(board["shots"])
        LOGGER.info("工作目录: %s", project)

        try:
            self.post(lambda: self._progress(0, "开始生成分镜图像..."))
            shots = []

            for shot, entry in zip(board["shots"], self.prompts):
                if self.cancel.is_set():
                    LOGGER.info("用户取消，已完成的镜头保留")
                    break
                done = self._render_shot(shot, entry, len(shots))
                if done:
                    shots.append(done)

            if not shots:
                LOGGER.error("没有任何镜头生成成功")
                self.post(lambda: self._progress(0, "生成失败"))
                return

            self.post(lambda: self._progress(92, "拼接成片..."))
            final = assemble.merge([s["video_url"] for s in shots], project, assemble.safe_name(poem))
            if final:
                self.assets["final"] = final
                self.post(lambda: self._finish(final))
                self.post(lambda: self._progress(100, "处理完成"))
                LOGGER.info("成片完成: %s", final)
            else:
                LOGGER.error("拼接失败")
                self.post(lambda: self._progress(0, "拼接失败"))
        except Exception:
            LOGGER.exception("生成过程异常")
        finally:
            self.busy = False
            self.post(lambda: self._set(generate_btn=tk.NORMAL, cancel_btn=tk.DISABLED,
                                        analyze_btn=tk.NORMAL))

    def _render_shot(self, shot, entry, done_count):
        """生成一镜的图像与片段，并在每一步之后做验证与自我修正。"""
        index = shot["index"]
        base = 10 + (done_count / max(self._total, 1)) * 78

        image_url, image_path, image_ok = self._render_image(shot, entry)
        if not image_ok:
            LOGGER.error("第 %d 镜图像生成失败，跳过该镜", index)
            return None
        self.post(lambda p=base: self._progress(p, f"第 {index} 镜图像完成"))

        video_url, clip_path = self._render_clip(shot, entry, image_url, image_path)
        if not video_url:
            LOGGER.error("第 %d 镜视频生成失败，跳过该镜", index)
            return None

        self.assets["images"].append(image_url)
        self.assets["clips"].append(video_url)
        self.post(lambda i=index: self._on_shot_done(i))
        return {"index": index, "image_url": image_url, "video_url": video_url,
                "image_path": image_path, "clip_path": clip_path}

    def _render_image(self, shot, entry):
        """生成一镜的图像，不达标时按修正要求重写提示词再试一次。

        返回 (图片地址, 本地路径, 是否拿到可用图)。走到最后一轮仍不达标时
        接受结果并记警告：继续重试只会持续消耗接口额度。
        """
        attempts = 1 + REPAIR_ROUNDS
        current = shot

        for round_no in range(attempts):
            prompt = entry["image"] if round_no == 0 else prompt_builder.image_prompt(current, self.board)
            LOGGER.info("第 %d 镜图像生成（第 %d 次）", shot["index"], round_no + 1)
            url = self.client.text_to_image(prompt)
            if not url:
                return None, None, False

            path = self._images_dir / f"第{shot['index']}镜.jpg"
            if not assemble.fetch(url, path):
                return None, None, False

            verdict = verify.check_image(path, f"第{shot['index']}镜图")
            if verdict.status != "fail" or round_no == attempts - 1:
                if verdict.status == "fail":
                    LOGGER.warning("第 %d 镜图像仍不达标: %s", shot["index"], verdict.detail)
                return url, str(path), True

            fix = prompt_builder.corrective_clause([verdict])
            LOGGER.warning("第 %d 镜图像不达标（%s），按提示修正后重生成", shot["index"], verdict.detail)
            current = prompt_builder.apply_correction(shot, fix)
            entry["image"] = prompt_builder.image_prompt(current, self.board)
        return None, None, False

    def _render_clip(self, shot, entry, image_url, image_path):
        """生成一镜的视频片段，并在运动或首帧一致性不达标时修正重生成。

        修正时必须连同输入图一起换掉：图生视频接口不接收文字形式的运动描述，
        沿用原图重试只会得到同样的静止画面。
        """
        attempts = 1 + REPAIR_ROUNDS
        current = shot

        for round_no in range(attempts):
            video_url = self.client.image_to_video(image_url, motion_prompt=entry["video"])
            if not video_url:
                return None, None

            clip_path = self._clips_dir / f"第{shot['index']}镜.mp4"
            if not assemble.fetch(video_url, clip_path):
                return None, None

            problems = [verify.check_motion(clip_path),
                        verify.check_image_against_clip(image_path, clip_path)]
            failed = [c for c in problems if c.status == "fail"]
            for check in problems:
                LOGGER.info("第 %d 镜验证 | %s", shot["index"], check)

            if not failed or round_no == attempts - 1:
                return video_url, str(clip_path)

            fix = prompt_builder.corrective_clause(failed)
            if not fix:
                return video_url, str(clip_path)

            LOGGER.warning("第 %d 镜视频不达标，按提示修正后重生成：%s", shot["index"], fix)
            current = prompt_builder.apply_correction(shot, fix)
            entry["video"] = prompt_builder.video_prompt(current, self.board)
            # 修正必须同时落回图像提示词：图生视频接口不接收运动描述，
            # 能改变成片运动的唯一杠杆就是换一张构图不同的输入图。
            entry["image"] = prompt_builder.image_prompt(current, self.board)
            image_url = self.client.text_to_image(entry["image"]) or image_url
        return None, None

    def _on_shot_done(self, index):
        """主线程：把完成的镜号写进两个列表并放开对应的下载播放按钮。"""
        self.image_list.insert(tk.END, f"第 {index} 镜图像")
        self.clip_list.insert(tk.END, f"第 {index} 镜片段")
        self._set(image_save_btn=tk.NORMAL, image_view_btn=tk.NORMAL,
                  clip_save_btn=tk.NORMAL, clip_play_btn=tk.NORMAL)

    def _finish(self, final):
        """主线程：显示成片路径，并放开播放与下载按钮。"""
        self.final_entry.config(state="normal")
        self.final_entry.delete(0, tk.END)
        self.final_entry.insert(0, final)
        self.final_entry.config(state="readonly")
        self._set(final_save_btn=tk.NORMAL, final_play_btn=tk.NORMAL,
                  play_btn=tk.NORMAL, pause_btn=tk.NORMAL, stop_btn=tk.NORMAL)

    def _progress(self, value, text):
        """更新进度条与当前步骤文字。"""
        self.progress["value"] = value
        self.progress_text.config(text=text)

    def stop(self):
        """请求取消生成。只置标志位，正在进行的镜头会完整走完再退出循环。"""
        self.cancel.set()
        self._playing = False
        LOGGER.info("用户取消了生成过程")
        self._progress(0, "已取消")
        self._set(cancel_btn=tk.DISABLED, analyze_btn=tk.NORMAL, generate_btn=tk.NORMAL)

    def _thread(self, target, *args):
        """起一个守护线程跑耗时任务。守护属性保证关窗时不会卡住进程退出。"""
        threading.Thread(target=target, args=args, daemon=True).start()

    # 播放

    def play(self):
        """从头播放成片。已在播放时不重复起线程。"""
        if not self.assets["final"]:
            messagebox.showerror("错误", "还没有成片")
            return
        if self._playing:
            return
        self._playing = True
        self._thread(self._playback, self.assets["final"])

    def _playback(self, path):
        """工作线程只解码，画面交给主线程渲染。单槽丢帧，不排队。"""
        capture = cv2.VideoCapture(path)
        self._cap = capture
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 24
            interval = 1.0 / max(fps, 1)
            while self._playing:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).resize(PREVIEW_SIZE, Image.LANCZOS)
                with self._frame_lock:
                    self._frame = image
                time.sleep(interval)
        except Exception:
            LOGGER.exception("播放异常")
        finally:
            capture.release()
            self._playing = False

    def pause(self):
        """暂停播放。播放线程仍在运行，下次 play 从当前位置继续。"""
        self._playing = False

    def halt(self):
        """停止播放并释放解码器，画面区恢复为提示文字。"""
        self._playing = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.screen.config(image="", text="已停止")
        self.screen.image = None

    def _open_external(self, target):
        """用系统默认播放器打开片段或成片。

        片段与成片不在界面内播放：界面里的播放器只解码成片，
        为几秒的片段再走一遍解码显示没有意义，交给系统播放器更省事也更稳。
        """
        if str(target).startswith("http"):
            messagebox.showinfo("提示", f"远程地址，请先下载：\n{target}")
            return
        if not os.path.exists(target):
            messagebox.showerror("错误", f"文件不存在：\n{target}")
            return
        try:
            if os.name == "nt":
                os.startfile(target)
            else:
                subprocess.Popen(["xdg-open", target])
        except OSError as exc:
            messagebox.showerror("错误", f"无法打开播放器：{exc}")

    def play_clip(self):
        """播放列表中选中的片段。"""
        picked = self.clip_list.curselection()
        if not picked:
            messagebox.showinfo("提示", "请先选择片段")
            return
        if picked[0] < len(self.assets["clips"]):
            self._open_external(self.assets["clips"][picked[0]])

    def play_final(self):
        """播放成片。"""
        if self.assets["final"]:
            self._open_external(self.assets["final"])

    # 存取

    def view_image(self):
        """在独立窗口中预览选中的分镜图像。"""
        picked = self.image_list.curselection()
        if not picked:
            messagebox.showinfo("提示", "请先选择图像")
            return
        index = picked[0]
        path = self._image_path(index)
        if not path:
            return
        window = tk.Toplevel(self.root)
        window.title(f"第 {index + 1} 镜图像")
        window.geometry("860x620")
        try:
            image = Image.open(path)
            image.load()
            image = ImageOps.contain(image, (820, 560))
            photo = ImageTk.PhotoImage(image)
            label = ttk.Label(window, image=photo)
            label.image = photo
            label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            ttk.Label(window, text=path).pack(side=tk.BOTTOM, fill=tk.X, padx=8)
        except OSError as exc:
            ttk.Label(window, text=f"无法加载：{exc}").pack(fill=tk.BOTH, expand=True)

    def _image_path(self, index):
        """由镜号反推磁盘路径。

        生成时按「诗词内容 / 分镜图像 / 第N镜.jpg」落盘，
        这里用同样的规则还原，所以不需要额外维护一份路径清单。
        文件不存在时返回 None，由调用方决定提示方式。
        """
        project = settings.PROJECT_DIR / assemble.safe_name(self._poem_text())
        candidate = project / "分镜图像" / f"第{index + 1}镜.jpg"
        return str(candidate) if candidate.is_file() else None

    def save_image(self):
        """把选中的分镜图像另存到用户指定位置。"""
        picked = self.image_list.curselection()
        if not picked:
            messagebox.showinfo("提示", "请先选择图像")
            return
        path = self._image_path(picked[0])
        if path:
            self._copy_out(path, "图像", ".jpg")

    def save_clip(self):
        """把选中的视频片段另存到用户指定位置。"""
        picked = self.clip_list.curselection()
        if not picked:
            messagebox.showinfo("提示", "请先选择片段")
            return
        project = settings.PROJECT_DIR / assemble.safe_name(self._poem_text())
        path = project / "视频片段" / f"第{picked[0] + 1}镜.mp4"
        if path.is_file():
            self._copy_out(str(path), "片段", ".mp4")

    def save_final(self):
        """把成片另存到用户指定位置。"""
        if self.assets["final"]:
            self._copy_out(self.assets["final"], "成片", ".mp4")

    def _copy_out(self, source, label, suffix):
        """弹出保存对话框并复制文件。用户取消时不提示，避免多余的弹窗。"""
        target = filedialog.asksaveasfilename(title=f"保存{label}", defaultextension=suffix,
                                              filetypes=[("所有文件", "*.*")])
        if not target:
            return
        try:
            shutil.copyfile(source, target)
        except OSError as exc:
            messagebox.showerror("错误", f"保存失败：{exc}")
            return
        messagebox.showinfo("已保存", f"{label}已保存到：\n{target}")

    # 自定义关键词

    def custom_preview(self):
        """按关键词生成一张预览图。"""
        keyword = self.keyword.get().strip()
        if not keyword:
            messagebox.showerror("错误", "请输入关键词")
            return
        self.custom_status.config(text="正在生成...")
        self.custom_gen_btn.config(state=tk.DISABLED)
        self._thread(self._do_custom_preview, keyword)

    def _do_custom_preview(self, keyword):
        """生成并落盘预览图，随后交回主线程登记到列表。

        预览图存到 Custom/预览图 而不是临时目录：列表里保存着该路径供
        「下载选中图像」使用，写进随用随删的临时目录会让下载必然失败。
        """
        try:
            prompt = f"{keyword}，{'，'.join(settings.CONTINUITY_RULES[:2])}"
            url = self.client.text_to_image(prompt[:settings.PROMPT_LIMIT])
            if not url:
                self.post(lambda: self.custom_status.config(text="生成失败"))
                return
            folder = settings.CUSTOM_OUTPUT_DIR / "预览图"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{assemble.safe_name(keyword)}_{int(time.time())}.jpg"
            if not assemble.fetch(url, path):
                self.post(lambda: self.custom_status.config(text="下载失败"))
                return
            image = Image.open(path)
            image.load()
            image = ImageOps.contain(image, (620, 320))
            self.post(lambda: self._register_custom(image, url, str(path), keyword))
            self.post(lambda: self.custom_status.config(text="生成完成"))
        except Exception as exc:
            LOGGER.exception("自定义预览生成失败")
            self.post(lambda: self.custom_status.config(text=f"错误：{exc}"))
        finally:
            self.post(lambda: self.custom_gen_btn.config(state=tk.NORMAL))

    def _register_custom(self, image, url, path, keyword):
        """把新预览图登记进列表。PhotoImage 必须在主线程创建，故在此完成。"""
        photo = ImageTk.PhotoImage(image)
        self.custom_images.append({"url": url, "path": path, "keyword": keyword, "photo": photo})
        self.custom_image.config(image=photo, text="")
        self.custom_image.image = photo
        self.custom_list.insert(tk.END, f"{len(self.custom_images)}. {keyword[:18]}")
        self._set(custom_save_btn=tk.NORMAL, sequence_btn=tk.NORMAL, custom_video_btn=tk.NORMAL)

    def custom_save(self):
        """把选中的预览图另存到用户指定位置。"""
        picked = self.custom_list.curselection()
        if not picked:
            messagebox.showinfo("提示", "请先选择图像")
            return
        self._copy_out(self.custom_images[picked[0]]["path"], "图像", ".jpg")

    def custom_sequence(self):
        """把已有预览图逐个转成视频片段，不拼接。"""
        if not self.custom_images:
            messagebox.showinfo("提示", "请先生成预览图")
            return
        self.sequence_btn.config(state=tk.DISABLED)
        self._thread(self._do_custom_sequence)

    def _do_custom_sequence(self):
        """工作线程：逐个预览图转片段，不拼接。"""
        try:
            clips = self._custom_clips()
            count = len(clips)
            self.post(lambda: messagebox.showinfo("完成", f"已生成 {count} 个片段"))
        except Exception as exc:
            LOGGER.exception("序列生成失败")
            self.post(lambda: messagebox.showerror("错误", str(exc)))
        finally:
            self.post(lambda: self.sequence_btn.config(state=tk.NORMAL))

    def custom_video(self):
        """把已有预览图转成片段并拼接为一段完整视频。"""
        if not self.custom_images:
            messagebox.showinfo("提示", "请先生成预览图")
            return
        self.custom_video_btn.config(state=tk.DISABLED)
        self._thread(self._do_custom_video)

    def _do_custom_video(self):
        """工作线程：预览图转片段后拼接为一段完整视频。"""
        try:
            clips = self._custom_clips()
            if not clips:
                self.post(lambda: messagebox.showerror("错误", "没有可用片段"))
                return
            final = assemble.merge([c["url"] for c in clips], settings.CUSTOM_OUTPUT_DIR, "custom_video")
            self.post(lambda: messagebox.showinfo("完成", f"成片：\n{final}"))
        except Exception as exc:
            LOGGER.exception("自定义成片失败")
            self.post(lambda: messagebox.showerror("错误", str(exc)))
        finally:
            self.post(lambda: self.custom_video_btn.config(state=tk.NORMAL))

    def _custom_clips(self):
        """把列表里的预览图逐个转为视频片段，返回成功的部分。

        与诗词流程共用同一个接口客户端，因此同样享受缓存与限流；
        区别只是画面描述直接取关键词，不经过意象抽取与分镜规划。
        """
        folder = settings.CUSTOM_OUTPUT_DIR / "视频片段"
        folder.mkdir(parents=True, exist_ok=True)
        clips = []
        for order, item in enumerate(self.custom_images, 1):
            if self.cancel.is_set():
                break
            LOGGER.info("自定义序列 %d/%d", order, len(self.custom_images))
            url = self.client.image_to_video(item["url"], motion_prompt=item.get("keyword", ""))
            if not url:
                continue
            path = folder / f"片段{len(clips) + 1}.mp4"
            if assemble.fetch(url, path):
                clips.append({"url": url, "path": str(path)})
        return clips


def enable_dpi_awareness():
    """让窗口按物理像素渲染。

    不声明 DPI 感知时，Windows 会把整个窗口位图拉伸到 125%/150%，
    高分屏上文字发虚，截图尺寸也与实际不符。
    ctypes 各平台都能导入，只有 windll 是 Windows 专有，因此这里捕获属性错误。
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def main():
    """程序入口：DPI 感知 → 登录校验 → 主界面。

    登录不通过时直接返回，不创建主窗口，也不启动后台清理线程。
    """
    enable_dpi_awareness()
    settings.configure_logging()

    login_root = tk.Tk()
    username = ask_login(login_root)
    if not username:
        LOGGER.info("未通过登录校验，程序退出")
        return

    threading.Thread(target=assemble.sweep_temp, daemon=True).start()
    root = tk.Tk()
    T2VideoWindow(root, username=username)
    root.mainloop()


if __name__ == "__main__":
    main()
