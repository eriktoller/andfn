import logging

import pytest

from andfn import utils


class _NonStreamHandler(logging.Handler):
    """Simple handler used to verify non-StreamHandler behavior."""

    def emit(self, record):
        return None


@pytest.fixture(autouse=True)
def _isolate_root_handlers():
    """Keep root handlers isolated so tests do not leak logger state."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)


def test_configure_logging_calls_dict_config(monkeypatch):
    calls = []

    def _fake_dict_config(config):
        calls.append(config)

    monkeypatch.setattr(utils.logging.config, "dictConfig", _fake_dict_config)

    utils.configure_logging()

    assert len(calls) == 1
    assert calls[0] == utils.LOGGING_CONFIG


def test_set_log_level_updates_stream_handlers_only_by_default():
    root = logging.getLogger()
    stream_handler = logging.StreamHandler()
    non_stream_handler = _NonStreamHandler()
    stream_handler.setLevel(logging.WARNING)
    non_stream_handler.setLevel(logging.WARNING)
    root.addHandler(stream_handler)
    root.addHandler(non_stream_handler)

    utils.set_log_level("debug")

    assert stream_handler.level == logging.DEBUG
    assert non_stream_handler.level == logging.WARNING


def test_set_log_level_updates_all_handlers_when_requested():
    root = logging.getLogger()
    stream_handler = logging.StreamHandler()
    non_stream_handler = _NonStreamHandler()
    root.addHandler(stream_handler)
    root.addHandler(non_stream_handler)

    utils.set_log_level("error", all_handlers=True)

    assert stream_handler.level == logging.ERROR
    assert non_stream_handler.level == logging.ERROR


def test_set_log_level_invalid_level_raises():
    with pytest.raises(TypeError, match="Invalid log level"):
        utils.set_log_level("not-a-level")


def test_enable_file_logging_appends_log_extension_and_sets_level(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    utils.enable_file_logging(name="session", loglevel="info")

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
    ]

    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename.endswith("session.log")
    assert file_handlers[0].level == logging.INFO


def test_enable_file_logging_avoids_duplicate_file_handlers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    utils.enable_file_logging(name="dupe", loglevel="debug")
    utils.enable_file_logging(name="dupe", loglevel="debug")

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
    ]

    assert len(file_handlers) == 1


def test_enable_file_logging_invalid_level_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TypeError, match="Invalid log level"):
        utils.enable_file_logging(name="bad", loglevel="not-a-level")
