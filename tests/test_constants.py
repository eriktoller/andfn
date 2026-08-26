import pytest

from andfn.constants import Constants, load_yaml_config


def test_default_constants():
    c = Constants()

    assert c.constants["RHO"] == 1000.0
    assert c.constants["G"] == 9.81
    assert c.constants["MAX_ITERATIONS"] == 50
    assert c.constants["NCOEF"] == 5
    assert c.constants["NUM_THREADS"] == -1


def test_change_constant():
    c = Constants()

    c.change_constants(MAX_ITERATIONS=100)

    assert c.constants["MAX_ITERATIONS"] == 100


def test_change_multiple_constants():
    c = Constants()

    c.change_constants(
        MAX_ITERATIONS=100,
        DAMPING=0.7,
        NCOEF=10,
    )

    assert c.constants["MAX_ITERATIONS"] == 100
    assert c.constants["DAMPING"] == 0.7
    assert c.constants["NCOEF"] == 10


def test_unknown_constant_is_ignored():
    c = Constants()

    c.change_constants(NOT_A_CONSTANT=123)

    assert "NOT_A_CONSTANT" not in c.constants.dtype.names


@pytest.mark.parametrize("value", [0, -1, -5])
def test_num_threads_must_be_positive(value):
    c = Constants()

    with pytest.raises(ValueError):
        c.change_constants(NUM_THREADS=value)


from unittest.mock import patch


@patch("andfn_darcytools.constants.set_num_threads")
def test_num_threads_calls_numba(mock_threads):
    c = Constants()

    c.change_constants(NUM_THREADS=4)

    mock_threads.assert_called_once_with(4)


from unittest.mock import patch

import pytest


@patch("andfn_darcytools.element.MAX_NCOEF", 100)
def test_max_ncoef_limit():
    c = Constants()

    with pytest.raises(ValueError):
        c.change_constants(MAX_NCOEF=101)


@patch("andfn_darcytools.element.MAX_ELEMENTS", 100)
def test_max_elements_limit():
    c = Constants()

    with pytest.raises(ValueError):
        c.change_constants(MAX_ELEMENTS=101)


def test_load_yaml_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert load_yaml_config() is None


def test_load_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = tmp_path / ".andfn_config.yaml"
    config.write_text(
        """
MAX_ITERATIONS: 100
DAMPING: 0.7
"""
    )

    result = load_yaml_config()

    assert result["MAX_ITERATIONS"] == 100
    assert result["DAMPING"] == 0.7


def test_yaml_configuration_applied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".andfn_config.yaml").write_text(
        """
MAX_ITERATIONS: 123
"""
    )

    c = Constants()

    assert c.constants["MAX_ITERATIONS"] == 123


import logging

logger = logging.getLogger("andfn")


def test_change_log_level():
    Constants.change_log_level(logging.DEBUG)

    assert logger.level == logging.DEBUG
