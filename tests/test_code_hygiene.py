"""代码卫生检查。

把这次清理掉的几类问题固化成断言，避免以后重新引入：
    - 库代码里不该出现 print（GUI 程序用 pythonw 启动时控制台输出会直接丢失）
    - 不该有裸 except，也不该在函数体内部临时 import
    - 中文不该写成 \\uXXXX 转义，字符串里不该出现 "??" 这类编码损坏残留
    - 不该有 ========== 之类的分节横幅注释
    - 不该有未使用的 import
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "t2video"
MODULES = sorted(PACKAGE.glob("*.py"))

BANNER = re.compile(r"^\s*#\s*[=-]{4,}")
BARE_EXCEPT = re.compile(r"^\s*except\s*:")
PRINT_CALL = re.compile(r"(?<![\w.])print\(")
UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")


def source_of(path):
    return path.read_text(encoding="utf-8")


def test_package_is_not_empty():
    assert MODULES, "没找到任何模块，路径可能不对"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
class TestHygiene:
    def test_no_print_calls(self, path):
        offenders = [
            f"L{i}: {line.strip()}"
            for i, line in enumerate(source_of(path).splitlines(), 1)
            if PRINT_CALL.search(line)
        ]
        assert offenders == [], f"{path.name} 里还有 print:\n" + "\n".join(offenders)

    def test_no_bare_except(self, path):
        offenders = [
            f"L{i}: {line.strip()}"
            for i, line in enumerate(source_of(path).splitlines(), 1)
            if BARE_EXCEPT.match(line)
        ]
        assert offenders == [], f"{path.name} 里有裸 except:\n" + "\n".join(offenders)

    def test_no_unicode_escaped_chinese(self, path):
        offenders = [
            f"L{i}: {line.strip()[:70]}"
            for i, line in enumerate(source_of(path).splitlines(), 1)
            if UNICODE_ESCAPE.search(line)
        ]
        assert offenders == [], f"{path.name} 里有 \\uXXXX 转义:\n" + "\n".join(offenders)

    def test_no_section_banner_comments(self, path):
        offenders = [
            f"L{i}: {line.strip()}"
            for i, line in enumerate(source_of(path).splitlines(), 1)
            if BANNER.match(line)
        ]
        assert offenders == [], f"{path.name} 里还有横幅注释:\n" + "\n".join(offenders)

    def test_no_mojibake_strings(self, path):
        tree = ast.parse(source_of(path))
        offenders = [
            f"L{node.lineno}: {node.value!r}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "??" in node.value
        ]
        assert offenders == [], f"{path.name} 里有编码损坏的字符串:\n" + "\n".join(offenders)

    def test_no_function_level_imports(self, path):
        tree = ast.parse(source_of(path))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        offenders.append(f"L{child.lineno} in {node.name}()")
        assert offenders == [], f"{path.name} 里函数内还有 import:\n" + "\n".join(offenders)

    def test_no_unused_imports(self, path):
        source = source_of(path)
        tree = ast.parse(source)
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)

        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                target = node
                while isinstance(target, ast.Attribute):
                    target = target.value
                if isinstance(target, ast.Name):
                    used.add(target.id)

        unused = sorted(bound - used)
        assert unused == [], f"{path.name} 里有未使用的 import: {unused}"

    def test_no_duplicate_top_level_definitions(self, path):
        tree = ast.parse(source_of(path))
        seen = {}
        duplicates = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in seen:
                    duplicates.append(node.name)
                seen[node.name] = node.lineno
        assert duplicates == [], f"{path.name} 里重复定义: {duplicates}"
