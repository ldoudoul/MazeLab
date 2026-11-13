# -*- coding: utf-8 -*-
"""MazeLab 迷宫寻路实验室 - Python 单文件源码。

文件名沿用课程分组作业示例；实际主题为 BFS / DFS / A* 迷宫寻路。
运行：python turing_machine.py
测试：python turing_machine.py --self-test
仅依赖 Python 标准库；图形环境需支持 tkinter/Tcl/Tk。
"""
from __future__ import annotations

import ctypes
import heapq
import random
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

VERSION = "1.1.0"
BG, PANEL, FIELD = "#10141a", "#171d25", "#202832"
TEXT, MUTED, GREEN = "#e8eef5", "#8c9bad", "#4ce2bb"
WALL, EMPTY, FRONTIER, VISITED = "#3a4654", "#1b222b", "#266761", "#25434d"
PURPLE, AMBER = "#be9dff", "#f4bb62"
ALGORITHMS = ("BFS · 广度优先搜索", "DFS · 深度优先搜索", "A* · 启发式搜索")


@dataclass
class Board:
    width: int
    height: int
    walls: list[bool]
    start: int
    end: int

    @classmethod
    def blank(cls, width=31, height=21):
        return cls(width, height, [False] * (width * height), width + 1,
                   (height - 2) * width + width - 2)

    def neighbors(self, p):
        x, y = p % self.width, p // self.width
        if y > 0:
            yield p - self.width
        if x < self.width - 1:
            yield p + 1
        if y < self.height - 1:
            yield p + self.width
        if x > 0:
            yield p - 1

    @classmethod
    def generate(cls, width=31, height=21, seed=None):
        """随机回溯挖通所有奇数坐标房间，另加少量环路。"""
        if width < 5 or height < 5 or width % 2 == 0 or height % 2 == 0:
            raise ValueError("迷宫生成尺寸必须是至少为 5 的奇数。")
        rng = random.Random(seed)
        b = cls.blank(width, height)
        b.walls = [True] * (width * height)
        b.walls[b.start] = False
        stack = [b.start]
        while stack:
            p = stack[-1]
            x, y = p % width, p // width
            candidates = []
            for dx, dy in ((0, -2), (2, 0), (0, 2), (-2, 0)):
                nx, ny = x + dx, y + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1:
                    q = ny * width + nx
                    if b.walls[q]:
                        candidates.append(q)
            if not candidates:
                stack.pop()
                continue
            q = rng.choice(candidates)
            b.walls[q] = b.walls[(p + q) // 2] = False
            stack.append(q)
        for _ in range(width * height // 30):
            x, y = rng.randrange(1, width - 1), rng.randrange(1, height - 1)
            p = y * width + x
            if ((not b.walls[p - 1] and not b.walls[p + 1]) or
                    (not b.walls[p - width] and not b.walls[p + width])):
                b.walls[p] = False
        return b

    def save(self, path):
        lines = ["MAZELAB 1", f"{self.width} {self.height} {self.start} {self.end}"]
        for y in range(self.height):
            lines.append("".join("#" if self.walls[y * self.width + x] else "."
                                 for x in range(self.width)))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        if Path(path).stat().st_size > 100_000:
            raise ValueError("地图文件过大。")
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
        if len(lines) < 3 or lines[0] != "MAZELAB 1":
            raise ValueError("不是有效的 MazeLab 地图。")
        try:
            w, h, s, e = map(int, lines[1].split())
        except ValueError as exc:
            raise ValueError("地图参数无效。") from exc
        if not (5 <= w <= 81 and 5 <= h <= 61 and len(lines) == h + 2
                and 0 <= s < w * h and 0 <= e < w * h and s != e):
            raise ValueError("地图尺寸或起终点无效。")
        if any(len(row) != w or set(row) - {"#", "."} for row in lines[2:]):
            raise ValueError("地图行长度或字符无效。")
        b = cls(w, h, [c == "#" for row in lines[2:] for c in row], s, e)
        if b.walls[s] or b.walls[e]:
            raise ValueError("起终点不能位于墙内。")
        return b


@dataclass
class SearchResult:
    frames: list[tuple[int, list[int]]]
    path: list[int]
    milliseconds: float


def search(board, algorithm):
    """搜索结果与 GUI 分离。算法序号 0=BFS，1=DFS，2=A*。

    BFS 使用 deque，DFS 使用 list 栈，A* 使用 heapq 最小堆。
    A* 使用曼哈顿启发式；本应用仅允许四方向等权移动。
    """
    if algorithm not in (0, 1, 2):
        raise ValueError("未知算法。")
    begin = time.perf_counter()
    prev, distance = {board.start: None}, {board.start: 0}
    closed, frames, path = set(), [], []

    def heuristic(p):
        return abs(p % board.width - board.end % board.width) + abs(
            p // board.width - board.end // board.width)

    queue = deque([board.start])
    stack = [board.start]
    heap = [(heuristic(board.start), 0, board.start)]
    order = 0
    while queue if algorithm == 0 else stack if algorithm == 1 else heap:
        if algorithm == 0:
            p = queue.popleft()
        elif algorithm == 1:
            p = stack.pop()
        else:
            _, _, p = heapq.heappop(heap)
        if p in closed:
            continue
        closed.add(p)
        added = []
        if p == board.end:
            frames.append((p, added))
            q = p
            while q is not None:
                path.append(q)
                q = prev[q]
            path.reverse()
            break
        for q in board.neighbors(p):
            if board.walls[q] or q in closed:
                continue
            if algorithm == 2:
                candidate = distance[p] + 1
                if candidate >= distance.get(q, float("inf")):
                    continue
                first = q not in distance
                prev[q], distance[q] = p, candidate
                order += 1
                heapq.heappush(heap, (candidate + heuristic(q), order, q))
                if first:
                    added.append(q)
            elif q not in prev:
                prev[q], distance[q] = p, distance[p] + 1
                (queue if algorithm == 0 else stack).append(q)
                added.append(q)
        frames.append((p, added))
    return SearchResult(frames, path, (time.perf_counter() - begin) * 1000)


class MazeLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MazeLab · 迷宫寻路实验室")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1320, sw - 80), min(860, sh - 100)
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.minsize(1040, 700)
        self.configure(bg=BG)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.algorithm = tk.StringVar(value=ALGORITHMS[0])
        self.size_choice = tk.StringVar(value="标准 · 31 × 21")
        self.speed = tk.IntVar(value=45)
        self.tool, self.timer_id, self.running = 0, None, False
        self.result, self.index, self.path_index = None, 0, 0
        self.current, self.last_cell = -1, -1
        self.dragging = False
        self.board = Board.generate()
        self.states = [0] * len(self.board.walls)
        self.cells, self.cell_size, self.origin = [], 1, (0, 0)
        self._style()
        self._layout()
        self.bind("<space>", lambda e: self._shortcut(self.toggle))
        self.bind("<n>", lambda e: self._shortcut(self.single_step))
        self.bind("<r>", lambda e: self._shortcut(self.reset))
        self.bind("<ButtonRelease-1>", self._end_drag)
        self.bind("<ButtonRelease-3>", self._end_drag)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self._dark_title)
        self.reset()

    def _shortcut(self, action):
        if not isinstance(self.focus_get(), ttk.Combobox):
            action()
            return "break"

    def _dark_title(self):
        if sys.platform == "win32":
            try:
                handle = ctypes.windll.user32.GetParent(self.winfo_id())
                value = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(handle, 20, ctypes.byref(value), 4)
            except (OSError, AttributeError):
                pass

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=FIELD, background=FIELD,
                        foreground=TEXT, borderwidth=0, padding=7, arrowcolor=MUTED)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD)],
                  foreground=[("readonly", TEXT)], selectbackground=[("readonly", FIELD)],
                  selectforeground=[("readonly", TEXT)])

    def label(self, parent, text, size=10, color=TEXT, **kw):
        return tk.Label(parent, text=text, font=("Microsoft YaHei UI", size),
                        bg=parent.cget("bg"), fg=color, anchor="w", **kw)

    def button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                         bg=GREEN if primary else FIELD, fg=BG if primary else TEXT,
                         activebackground="#70eccd" if primary else "#303b48",
                         activeforeground=BG if primary else TEXT, cursor="hand2",
                         padx=12, pady=6, highlightthickness=0)

    def _layout(self):
        header = tk.Frame(self, bg=BG, padx=28, pady=14)
        header.pack(fill="x")
        self.label(header, "◈  MazeLab", 25, GREEN).pack(anchor="w")
        self.label(header, "迷宫寻路实验室  /  把算法变成看得见的思考", 10, MUTED).pack(anchor="w", pady=(4, 0))
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        side_shell = tk.Frame(body, bg=PANEL, width=265)
        side_shell.pack(side="left", fill="y")
        side_shell.pack_propagate(False)
        scroll = tk.Canvas(side_shell, bg=PANEL, width=1, highlightthickness=0)
        scroll.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(side_shell, orient="vertical", command=scroll.yview)
        bar.pack(side="right", fill="y")
        scroll.configure(yscrollcommand=bar.set)
        side = tk.Frame(scroll, bg=PANEL, padx=18, pady=10)
        window = scroll.create_window(0, 0, window=side, anchor="nw")
        side.bind("<Configure>", lambda e: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda e: scroll.itemconfigure(window, width=e.width))

        def section(text):
            self.label(side, text, 10, MUTED).pack(fill="x", pady=(9, 7))

        section("01  /  搜索策略")
        self.algo_box = ttk.Combobox(side, values=ALGORITHMS, state="readonly", textvariable=self.algorithm)
        self.algo_box.pack(fill="x")
        self.algo_box.bind("<<ComboboxSelected>>", lambda e: self.change_algorithm())
        self.description = self.label(side, "", 9, MUTED, wraplength=215, justify="left")
        self.description.pack(fill="x", pady=(8, 6))
        section("02  /  迷宫画布")
        size_box = ttk.Combobox(side, values=("精巧 · 21 × 15", "标准 · 31 × 21", "宽阔 · 41 × 27"),
                                state="readonly", textvariable=self.size_choice)
        size_box.pack(fill="x")
        size_box.bind("<<ComboboxSelected>>", lambda e: self.generate())
        self.button(side, "生成新迷宫", self.generate).pack(fill="x", pady=(12, 8))
        pair = tk.Frame(side, bg=PANEL)
        pair.pack(fill="x")
        pair.columnconfigure((0, 1), weight=1)
        self.button(pair, "空白画布", self.blank).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.button(pair, "清除轨迹", self.reset).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        section("03  /  编辑工具")
        tool_frame = tk.Frame(side, bg=PANEL)
        tool_frame.pack(fill="x")
        tool_frame.columnconfigure((0, 1), weight=1)
        self.tool_buttons = []
        for i, name in enumerate(("画墙", "橡皮", "起点 S", "终点 E")):
            b = self.button(tool_frame, name, lambda k=i: self.choose_tool(k))
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 5) if i % 2 == 0 else (5, 0), pady=4)
            self.tool_buttons.append(b)
        self.choose_tool(0)
        self.label(side, "左键拖动绘制，右键快速擦除", 9, MUTED).pack(fill="x", pady=(10, 4))
        section("04  /  演示速度")
        tk.Scale(side, from_=1, to=100, orient="horizontal", variable=self.speed,
                 bg=PANEL, fg=MUTED, troughcolor=FIELD, activebackground=GREEN,
                 highlightthickness=0, bd=0, sliderrelief="flat", showvalue=False).pack(fill="x")
        file_frame = tk.Frame(side, bg=PANEL)
        file_frame.pack(fill="x", pady=(18, 10))
        file_frame.columnconfigure((0, 1), weight=1)
        self.button(file_frame, "导入地图", self.load_map).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.button(file_frame, "保存地图", self.save_map).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        legend = tk.Frame(right, bg=BG, padx=22, pady=14)
        legend.pack(fill="x")
        for name, color in (("起点", GREEN), ("终点", PURPLE), ("待探索", FRONTIER),
                            ("已探索", VISITED), ("路径", GREEN)):
            self.label(legend, "●", 10, color).pack(side="left")
            self.label(legend, name, 9, MUTED).pack(side="left", padx=(5, 22))
        controls = tk.Frame(right, bg=PANEL, padx=20, pady=16)
        controls.pack(side="bottom", fill="x")
        row = tk.Frame(controls, bg=PANEL)
        row.pack(fill="x")
        self.play = self.button(row, "开始探索 / Space", self.toggle, True)
        self.play.pack(side="left", padx=(0, 10))
        self.button(row, "单步 / N", self.single_step).pack(side="left", padx=(0, 10))
        self.button(row, "算法对比", self.compare).pack(side="left")
        self.metrics = self.label(row, "已探索 0  /  路径 —", 11)
        self.metrics.pack(side="right", padx=10)
        self.status = self.label(controls, "", 9, MUTED, wraplength=740, justify="left", height=3)
        self.status.pack(fill="x", pady=(12, 0))
        self.canvas = tk.Canvas(right, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", lambda e: self.edit(e, False))
        self.canvas.bind("<B1-Motion>", lambda e: self.edit(e, False))
        self.canvas.bind("<Button-3>", lambda e: self.edit(e, True))
        self.canvas.bind("<B3-Motion>", lambda e: self.edit(e, True))
        self.change_algorithm()

    def dimensions(self):
        return {"精巧 · 21 × 15": (21, 15), "标准 · 31 × 21": (31, 21),
                "宽阔 · 41 × 27": (41, 27)}[self.size_choice.get()]

    def generate(self):
        self.board = Board.generate(*self.dimensions())
        self.reset()

    def blank(self):
        self.board = Board.blank(*self.dimensions())
        self.reset()

    def stop(self):
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.running = False

    def reset(self):
        self.stop()
        self.result, self.index, self.path_index = None, 0, 0
        self.current = -1
        self.states = [0] * len(self.board.walls)
        self.metrics.config(text="已探索 0  /  路径 —")
        self.status.config(text="准备就绪。选择算法开始探索，或直接设计你的迷宫。\nSpace 播放/暂停    N 单步    R 清除轨迹；编辑前请清除轨迹。")
        self.update_play()
        self.draw()

    def change_algorithm(self):
        descriptions = ("像水波一样逐层向外扩散。\n保证最短路径。\n时间复杂度 O(V + E)",
                        "沿着一条路深入，遇到尽头再返回。\n能找到路径，不保证最短。\n时间复杂度 O(V + E)",
                        "优先探索更接近目标的方向。\n曼哈顿启发式，保证最短路径。\n优先级 f = g + h")
        self.description.config(text=descriptions[ALGORITHMS.index(self.algorithm.get())])
        self.reset()

    def choose_tool(self, tool):
        self.tool = tool
        for i, b in enumerate(self.tool_buttons):
            b.configure(bg="#234944" if i == tool else FIELD,
                        fg=GREEN if i == tool else TEXT)

    def update_play(self):
        if self.running:
            text = "暂停探索 / Space"
        elif self.result is None:
            text = "开始探索 / Space"
        elif self.finished():
            text = "重新探索 / Space"
        else:
            text = "继续探索 / Space"
        self.play.configure(text=text)

    def finished(self):
        return (self.result is not None and self.index >= len(self.result.frames)
                and self.path_index >= len(self.result.path))

    def ensure(self):
        if self.result is None:
            self.result = search(self.board, ALGORITHMS.index(self.algorithm.get()))
            self.states[self.board.start] = 1

    def toggle(self):
        if self.running:
            self.stop()
        else:
            if self.finished():
                self.reset()
            self.ensure()
            self.running = True
            self.tick()
        self.update_play()

    def single_step(self):
        self.stop()
        self.ensure()
        self.advance()
        self.update_play()
        self.draw()

    def tick(self):
        self.timer_id = None
        for _ in range(1 + self.speed.get() // 12):
            if not self.running:
                break
            self.advance()
        self.draw()
        if self.running:
            self.timer_id = self.after(30, self.tick)
        self.update_play()

    def advance(self):
        r = self.result
        if self.index < len(r.frames):
            p, added = r.frames[self.index]
            self.index += 1
            self.current, self.states[p] = p, 2
            for q in added:
                self.states[q] = 1
            self.status.config(text=f"正在探索 ({p % self.board.width}, {p // self.board.width})，发现 {len(added)} 个新节点。\n青色为待探索节点，蓝灰色为已探索节点，金色为当前节点。")
        elif self.path_index < len(r.path):
            self.current = -1
            self.states[r.path[self.path_index]] = 3
            self.path_index += 1
            self.status.config(text="已找到终点，正在回放起点到终点的路径。")
        if self.finished():
            self.stop()
            self.current = -1
            if r.path:
                guarantee = "DFS 不保证最短。" if ALGORITHMS.index(self.algorithm.get()) == 1 else "当前四方向等权网格的最短路径。"
                self.status.config(text=f"探索完成 · 路径长度 {len(r.path)-1} 步。{guarantee}\n纯算法耗时 {r.milliseconds:.3f} ms（不含动画；单次计时仅供参考）。")
            else:
                self.status.config(text="未找到路径：起点与终点不连通。\n清除轨迹后可以修改墙壁、起终点，或重新生成迷宫。")
        length = str(max(0, self.path_index - 1)) if self.path_index else "—"
        self.metrics.config(text=f"已探索 {self.index}  /  路径 {length}")

    def draw(self):
        if not hasattr(self, "canvas"):
            return
        c, b = self.canvas, self.board
        width, height = c.winfo_width(), c.winfo_height()
        self.cell_size = cell = max(1, min((width - 24) / b.width, (height - 24) / b.height))
        self.origin = (ox, oy) = ((width - cell * b.width) / 2, (height - cell * b.height) / 2)
        c.delete("all")
        c.create_rectangle(ox - 9, oy - 9, ox + cell * b.width + 9,
                           oy + cell * b.height + 9, fill="#141920", outline="")
        for p, wall in enumerate(b.walls):
            color = WALL if wall else EMPTY
            if self.states[p]:
                color = {1: FRONTIER, 2: VISITED, 3: GREEN}[self.states[p]]
            if p == self.current:
                color = AMBER
            if p == b.start:
                color = GREEN
            elif p == b.end:
                color = PURPLE
            x, y = ox + p % b.width * cell, oy + p // b.width * cell
            c.create_rectangle(x + 1, y + 1, x + cell - 1, y + cell - 1, fill=color, outline="")
            if p in (b.start, b.end) and cell >= 14:
                c.create_text(x + cell / 2, y + cell / 2, text="S" if p == b.start else "E",
                              fill=BG, font=("Segoe UI", max(7, int(cell * .34)), "bold"))

    def edit(self, event, erase):
        if self.result is not None:
            return
        ox, oy = self.origin
        x, y = int((event.x - ox) // self.cell_size), int((event.y - oy) // self.cell_size)
        if not (0 <= x < self.board.width and 0 <= y < self.board.height):
            return
        p, b = y * self.board.width + x, self.board
        if p == self.last_cell:
            return
        self.last_cell = p
        if erase or self.tool == 1:
            if p not in (b.start, b.end):
                b.walls[p] = False
        elif self.tool == 2 and p != b.end:
            b.start, b.walls[p] = p, False
        elif self.tool == 3 and p != b.start:
            b.end, b.walls[p] = p, False
        elif self.tool == 0 and p not in (b.start, b.end):
            b.walls[p] = True
        self.draw()

    def _end_drag(self, event):
        self.last_cell = -1

    def save_map(self):
        self.stop()
        self.update_play()
        path = filedialog.asksaveasfilename(parent=self, title="保存迷宫", defaultextension=".maze",
                                          initialfile="我的迷宫.maze", filetypes=[("MazeLab 地图", "*.maze")])
        if path:
            try:
                self.board.save(path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("保存失败", str(exc), parent=self)

    def load_map(self):
        self.stop()
        self.update_play()
        path = filedialog.askopenfilename(parent=self, title="导入迷宫", filetypes=[("MazeLab 地图", "*.maze")])
        if path:
            try:
                board = Board.load(path)
            except (OSError, ValueError, UnicodeError) as exc:
                messagebox.showerror("导入失败", str(exc), parent=self)
                return
            self.board = board
            self.reset()

    def compare(self):
        self.stop()
        self.update_play()
        win = tk.Toplevel(self, bg=PANEL)
        win.title("同一地图 · 算法对比")
        win.geometry("650x340")
        win.resizable(False, False)
        win.transient(self)
        self.label(win, "同一张地图，三种思考方式", 18).pack(anchor="w", padx=26, pady=(22, 16))
        table = tk.Frame(win, bg=PANEL)
        table.pack(fill="x", padx=28)
        for col, name in enumerate(("算法", "探索节点", "路径步数", "最短保证")):
            table.columnconfigure(col, weight=1)
            self.label(table, name, 10, MUTED).grid(row=0, column=col, sticky="w", pady=8)
        for row in range(3):
            result = search(self.board, row)
            values = (("BFS", "DFS", "A*")[row], str(len(result.frames)),
                      str(len(result.path) - 1) if result.path else "无路径", "否" if row == 1 else "是")
            for col, value in enumerate(values):
                self.label(table, value, 11, GREEN if col == 0 else TEXT).grid(row=row + 1, column=col, sticky="w", pady=10)
        self.label(win, "相同地图与起终点；路径按移动步数计算。\n最短保证适用于四方向等权网格。", 9, MUTED, justify="left").pack(anchor="w", padx=28, pady=16)
        win.bind("<Escape>", lambda e: win.destroy())
        win.grab_set()
        win.focus_set()

    def close(self):
        self.stop()
        self.destroy()


def self_test():
    """可复现测试；失败抛出 AssertionError，并返回非零退出码。"""
    checks = 0

    def check(ok, reason):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(reason)

    def valid(board, result):
        if not result.path:
            return
        check(result.path[0] == board.start and result.path[-1] == board.end, "起终点")
        check(len({p for p, _ in result.frames}) == len(result.frames), "重复扩展")
        for i, p in enumerate(result.path):
            check(not board.walls[p], "路径穿墙")
            if i:
                check(p in board.neighbors(result.path[i - 1]), "路径不连续")

    b = Board.blank(9, 9)
    for a in range(3):
        r = search(b, a)
        valid(b, r)
        if a != 1:
            check(len(r.path) - 1 == 12, "空白网格最短路径")
    for seed in range(120):
        w, h = (31, 21) if seed % 2 == 0 else (21, 15)
        b = Board.generate(w, h, seed)
        results = [search(b, a) for a in range(3)]
        check(all(r.path for r in results), "随机迷宫必须可达")
        check(len(results[0].path) == len(results[2].path), "BFS/A* 最短长度一致")
        for result in results:
            valid(b, result)
    b = Board.blank(9, 9)
    for q in b.neighbors(b.end):
        b.walls[q] = True
    for a in range(3):
        check(not search(b, a).path, "无解处理")
    with tempfile.TemporaryDirectory(prefix="MazeLab-test-") as tmp:
        path = Path(tmp) / "迷宫.maze"
        b = Board.generate(seed=42)
        b.save(path)
        check(Board.load(path) == b, "文件往返")
        for text in ("wrong", "MAZELAB 1\n999999 21 32 618\n#",
                     "MAZELAB 1\n5 5 6 6\n" + ".....\n" * 5,
                     "MAZELAB 1\n5 5 6 18\n" + "?????\n" * 5):
            path.write_text(text, encoding="utf-8")
            rejected = False
            try:
                Board.load(path)
            except ValueError:
                rejected = True
            check(rejected, "非法地图必须拒绝")
    print(f"PASS: {checks} assertions; 120 seeded mazes; BFS/DFS/A*; unreachable; file roundtrip; 4 invalid files.")
    return checks


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        MazeLab().mainloop()
