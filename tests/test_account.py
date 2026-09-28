"""本地账户与口令校验测试。

口令闸门是真实校验，因此这里断言的是密码学行为：
同一口令可以稳定通过、错误口令一定失败、文件里不出现明文。
"""

import json

import pytest

from t2video import account


@pytest.fixture
def store(tmp_path):
    return tmp_path / "account.json"


class TestCreate:
    def test_creates_file(self, store):
        account.create("张三", "secret123", path=store)
        assert store.is_file()

    def test_returns_record(self, store):
        record = account.create("张三", "secret123", path=store)
        assert record["username"] == "张三"
        assert record["created"]

    def test_password_never_stored_in_plaintext(self, store):
        """核心断言：文件里不得出现口令明文。"""
        account.create("张三", "MyS3cretPass", path=store)
        raw = store.read_text(encoding="utf-8")
        assert "MyS3cretPass" not in raw
        assert "digest" in raw and "salt" in raw

    def test_salt_differs_between_accounts(self, tmp_path):
        first = account.create("a", "samepassword", path=tmp_path / "a.json")
        second = account.create("b", "samepassword", path=tmp_path / "b.json")
        assert first["salt"] != second["salt"]
        assert first["digest"] != second["digest"]

    def test_rejects_short_password(self, store):
        with pytest.raises(account.AccountError):
            account.create("张三", "123", path=store)

    def test_rejects_empty_username(self, store):
        with pytest.raises(account.AccountError):
            account.create("   ", "secret123", path=store)

    def test_overwrites_existing(self, store):
        account.create("张三", "firstpass", path=store)
        account.create("李四", "secondpass", path=store)
        assert account.load(store)["username"] == "李四"
        assert account.verify("李四", "secondpass", path=store)
        assert not account.verify("张三", "firstpass", path=store)

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "config" / "deep" / "account.json"
        account.create("张三", "secret123", path=nested)
        assert nested.is_file()


class TestLoad:
    def test_missing_returns_none(self, store):
        assert account.load(store) is None

    def test_reads_record(self, store):
        account.create("张三", "secret123", path=store)
        assert account.load(store)["username"] == "张三"

    def test_corrupt_json_returns_none(self, store):
        store.write_text("{ 这不是 JSON", encoding="utf-8")
        assert account.load(store) is None

    def test_incomplete_record_returns_none(self, store):
        store.write_text(json.dumps({"username": "张三"}), encoding="utf-8")
        assert account.load(store) is None

    def test_exists_reflects_file(self, store):
        assert not account.exists(store)
        account.create("张三", "secret123", path=store)
        assert account.exists(store)


class TestVerify:
    def test_correct_password_passes(self, store):
        account.create("张三", "secret123", path=store)
        assert account.verify("张三", "secret123", path=store) is True

    def test_wrong_password_fails(self, store):
        account.create("张三", "secret123", path=store)
        assert account.verify("张三", "secret124", path=store) is False

    def test_wrong_username_fails(self, store):
        account.create("张三", "secret123", path=store)
        assert account.verify("李四", "secret123", path=store) is False

    def test_empty_password_fails(self, store):
        account.create("张三", "secret123", path=store)
        assert account.verify("张三", "", path=store) is False

    def test_username_is_trimmed(self, store):
        account.create("张三", "secret123", path=store)
        assert account.verify("  张三  ", "secret123", path=store) is True

    def test_case_sensitive_password(self, store):
        account.create("张三", "Secret123", path=store)
        assert account.verify("张三", "secret123", path=store) is False

    def test_without_account_raises(self, store):
        with pytest.raises(account.AccountError):
            account.verify("张三", "secret123", path=store)

    def test_repeated_verification_is_stable(self, store):
        account.create("张三", "secret123", path=store)
        assert [account.verify("张三", "secret123", path=store) for _ in range(3)] == \
            [True, True, True]

    def test_custom_iterations_round_trip(self, store):
        """迭代次数写在记录里，校验时按记录取值，改配置后旧账户仍可用。"""
        account.create("张三", "secret123", path=store, iterations=1000)
        assert account.load(store)["iterations"] == 1000
        assert account.verify("张三", "secret123", path=store) is True


class TestReset:
    def test_removes_file(self, store):
        account.create("张三", "secret123", path=store)
        assert account.reset(store) is True
        assert not store.is_file()
        assert account.load(store) is None

    def test_reset_missing_is_false(self, store):
        assert account.reset(store) is False

    def test_can_recreate_after_reset(self, store):
        account.create("张三", "secret123", path=store)
        account.reset(store)
        account.create("李四", "newpass123", path=store)
        assert account.verify("李四", "newpass123", path=store)


class TestPasswordLengthBoundary:
    def test_minimum_length_accepted(self, store):
        password = "x" * account.MIN_PASSWORD_LENGTH
        account.create("张三", password, path=store)
        assert account.verify("张三", password, path=store) is True

    def test_one_below_minimum_rejected(self, store):
        with pytest.raises(account.AccountError):
            account.create("张三", "x" * (account.MIN_PASSWORD_LENGTH - 1), path=store)
