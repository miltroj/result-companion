from pathlib import Path

from result_companion.core.rc_paths import BUNDLED_DEFAULT_CONFIG, ensure_default_config


def test_ensure_default_config_copies_bundled_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr("result_companion.core.rc_paths.RC_USER_DIR", tmp_path)

    result = ensure_default_config()

    assert result == tmp_path / "default_config.yaml"
    assert result.exists()
    assert result.read_text() == BUNDLED_DEFAULT_CONFIG.read_text()


def test_ensure_default_config_reuses_existing(tmp_path, monkeypatch):
    monkeypatch.setattr("result_companion.core.rc_paths.RC_USER_DIR", tmp_path)
    existing = tmp_path / "default_config.yaml"
    existing.write_text("custom: true")

    result = ensure_default_config()

    assert result.read_text() == "custom: true"


def test_ensure_default_config_creates_nested_dir(tmp_path, monkeypatch):
    nested = tmp_path / "deep" / "nested"
    monkeypatch.setattr("result_companion.core.rc_paths.RC_USER_DIR", nested)

    result = ensure_default_config()

    assert nested.exists()
    assert result.exists()


def test_ensure_default_config_falls_back_to_bundled_when_root(tmp_path, monkeypatch):
    monkeypatch.setattr("result_companion.core.rc_paths.RC_USER_DIR", tmp_path)
    monkeypatch.setattr("result_companion.core.rc_paths._is_running_as_root", lambda: True)

    result = ensure_default_config()

    assert result == BUNDLED_DEFAULT_CONFIG
    assert not (tmp_path / "default_config.yaml").exists()


def test_bundled_default_config_exists():
    assert BUNDLED_DEFAULT_CONFIG.exists(), (
        f"Bundled config missing at {BUNDLED_DEFAULT_CONFIG}"
    )
