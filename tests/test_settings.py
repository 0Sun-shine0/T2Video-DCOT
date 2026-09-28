"""密钥读取测试。

回归用例针对一个真实发生过的问题：.env 里用的是 VOLC_ACCESS_KEY /
VOLC_SECRET_KEY，而代码只读 ACCESS_KEY / SECRET_KEY，
结果程序能启动、界面正常，但一提交生成就报缺少密钥。
"""

import importlib

import dotenv
import pytest

import t2video.settings as settings


def reload_settings(monkeypatch, env):
    """在受控环境下重载 settings。

    必须屏蔽 load_dotenv：否则重载时它会重新读取真实的 .env，
    测试就变成在断言真实密钥，既不可靠也会把密钥带进断言输出。
    """
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    for name in ("ACCESS_KEY", "SECRET_KEY", "VOLC_ACCESS_KEY",
                 "VOLC_SECRET_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(settings)


class TestEnvNames:
    def test_reads_plain_names(self, monkeypatch):
        module = reload_settings(monkeypatch, {
            "ACCESS_KEY": "ak-plain", "SECRET_KEY": "sk-plain",
        })
        assert module.ACCESS_KEY == "ak-plain"
        assert module.SECRET_KEY == "sk-plain"

    def test_reads_volc_prefixed_names(self, monkeypatch):
        """回归：控制台示例用 VOLC_ 前缀，必须也能识别。"""
        module = reload_settings(monkeypatch, {
            "VOLC_ACCESS_KEY": "ak-volc", "VOLC_SECRET_KEY": "sk-volc",
        })
        assert module.ACCESS_KEY == "ak-volc"
        assert module.SECRET_KEY == "sk-volc"

    def test_plain_name_wins_when_both_present(self, monkeypatch):
        module = reload_settings(monkeypatch, {
            "ACCESS_KEY": "ak-plain", "VOLC_ACCESS_KEY": "ak-volc",
        })
        assert module.ACCESS_KEY == "ak-plain"

    def test_empty_value_falls_through(self, monkeypatch):
        module = reload_settings(monkeypatch, {
            "ACCESS_KEY": "   ", "VOLC_ACCESS_KEY": "ak-volc",
        })
        assert module.ACCESS_KEY == "ak-volc"

    def test_missing_everything_is_empty(self, monkeypatch):
        module = reload_settings(monkeypatch, {})
        assert module.ACCESS_KEY == ""
        assert module.SECRET_KEY == ""
        assert module.DEEPSEEK_KEY == ""

    def test_deepseek_key(self, monkeypatch):
        module = reload_settings(monkeypatch, {"DEEPSEEK_API_KEY": "ds-key"})
        assert module.DEEPSEEK_KEY == "ds-key"

    def test_values_are_stripped(self, monkeypatch):
        module = reload_settings(monkeypatch, {"ACCESS_KEY": "  ak  "})
        assert module.ACCESS_KEY == "ak"

    def test_hint_lists_both_spellings(self, monkeypatch):
        module = reload_settings(monkeypatch, {})
        assert "ACCESS_KEY" in module.CREDENTIAL_HINT
        assert "VOLC_ACCESS_KEY" in module.CREDENTIAL_HINT
        assert "DEEPSEEK_API_KEY" in module.CREDENTIAL_HINT

    def test_restored_after_test(self):
        """重载不应污染后续测试：本模块结束后配置应恢复为原始状态。"""
        assert settings.API_HOST == "visual.volcengineapi.com"
