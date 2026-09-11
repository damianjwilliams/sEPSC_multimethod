"""Tests that a config file in effect is never silent.

Why this file exists
--------------------
A `multimethod.config.json` written by `calibrate --write` sat beside the
package and shaped every later run -- a 4-sigma amplitude floor that changed
one detector's event count by 11% -- and no run said so. The parameters were in
each run's sidecar, so nothing was unrecoverable, but recoverable afterwards is
not the same as visible at the time.

So these tests are mostly about OUTPUT, not about computation: that the message
appears, that it names the fields, and that the commands which load settings
actually print it. The last part matters because the failure mode here is a
command that quietly works while omitting the one line the user needed.
"""

from __future__ import annotations

import ast
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import cli, detectors
from multimethod.settings import (ConfigSource, Settings, default_config_path,
                                  describe_source, load_with_source)

HERE = os.path.dirname(os.path.abspath(__file__))
CLI_PY = os.path.join(HERE, os.pardir, "multimethod", "cli.py")


# ------------------------------------------------------------ what it detects

def test_no_config_file_reports_the_built_in_defaults(tmp_path):
    missing = str(tmp_path / "nope.json")
    s, src = load_with_source(missing)
    assert not src.exists
    assert not src.in_effect
    assert src.overrides == ()
    assert "built-in defaults" in src.describe()
    assert s.to_dict() == Settings().to_dict()


def test_a_config_matching_the_defaults_is_reported_as_inert(tmp_path):
    """Worth saying explicitly: the file exists but changes nothing. Otherwise
    a user deletes it looking for a difference that was never there."""
    path = Settings().save(str(tmp_path / "cfg.json"))
    _s, src = load_with_source(path)
    assert src.exists
    assert src.overrides == ()
    assert not src.in_effect, "a file that changes nothing is not 'in effect'"
    assert "matches the built-in defaults" in src.describe()


def test_every_changed_field_is_named_with_both_values(tmp_path):
    s = Settings()
    s.tolerance_ms = 0.5
    s.set_params_for("deconvolution", type(s.params_for("deconvolution"))(
        **{**detectors.params_as_dict(s.params_for("deconvolution")),
           "min_amp_abs": 11.13}))
    path = s.save(str(tmp_path / "cfg.json"))

    _back, src = load_with_source(path)
    keys = {k for k, _, _ in src.overrides}
    assert "tolerance_ms" in keys
    assert "detectors.deconvolution.min_amp_abs" in keys, (
        "a nested detector parameter must be reported by its dotted path, not "
        "hidden inside a section")
    assert src.in_effect

    text = src.describe()
    assert "11.13" in text and "0.5" in text
    assert "3.0" in text, "the built-in value it replaced must be shown too"
    assert str(len(src.overrides)) in text


def test_a_field_the_config_omits_is_not_reported_as_a_difference(tmp_path):
    """`from_dict` is forgiving, so a partial config is normal. Only genuine
    differences should be listed, or the message is noise."""
    (tmp_path / "cfg.json").write_text(json.dumps({"tolerance_ms": 0.5}))
    _s, src = load_with_source(str(tmp_path / "cfg.json"))
    assert [k for k, _, _ in src.overrides] == ["tolerance_ms"]


# ------------------------------------------------- explicit vs picked up

def test_an_explicitly_named_config_is_not_warned_about(tmp_path):
    s = Settings()
    s.tolerance_ms = 0.5
    path = s.save(str(tmp_path / "cfg.json"))
    _back, src = load_with_source(path)
    assert src.explicit
    assert "picked up automatically" not in src.describe(), (
        "the user passed --config, so they already know")
    assert "--config" in src.describe()


def test_the_default_path_is_the_one_that_gets_the_warning():
    """This is the hazardous case: it applies to commands that never mention
    the file, which is exactly how it went unnoticed."""
    s = Settings()
    s.tolerance_ms = 0.5
    src = describe_source(s, default_config_path(), exists=True, explicit=False)
    assert "picked up automatically" in src.describe()
    assert "rename it" in src.describe()


def test_load_with_source_resolves_the_default_path_when_given_none(tmp_path):
    _s, src = load_with_source(None)
    assert src.path == default_config_path()
    assert src.explicit is False


# --------------------------------------------------------- recorded on disk

def test_the_source_is_json_serialisable_for_the_sidecar(tmp_path):
    s = Settings()
    s.tolerance_ms = 0.5
    path = s.save(str(tmp_path / "cfg.json"))
    _back, src = load_with_source(path)
    blob = json.dumps(src.as_dict())          # must not raise
    back = json.loads(blob)
    assert back["exists"] is True
    assert back["overrides"]["tolerance_ms"] == {"builtin": 3.0, "in_effect": 0.5}


def test_the_run_sidecar_records_where_the_settings_came_from(tmp_path,
                                                              monkeypatch):
    """`settings` in the sidecar says what the values were. It does not say
    that they were not the defaults, or which file supplied them."""
    from multimethod import runner
    from test_runner_and_settings import synth_trace

    trace, _ = synth_trace(duration_s=8.0, seed=3)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)

    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))
    loaded, src = load_with_source(cfg)

    out = runner.run_all(str(tmp_path / "rec.abf"), 0, loaded,
                         methods=["threshold"], config_source=src)
    out.write()
    with open(out.stem + runner.RUN_SUFFIX, encoding="utf-8") as fh:
        sidecar = json.load(fh)
    assert sidecar["config_source"]["exists"] is True
    assert "tolerance_ms" in sidecar["config_source"]["overrides"]


def test_a_run_without_provenance_records_null_rather_than_failing(tmp_path,
                                                                  monkeypatch):
    """`run_all` is a library call as well as a CLI one, and a caller that
    never touched a config file must still be able to write results."""
    from multimethod import runner
    from test_runner_and_settings import synth_trace

    trace, _ = synth_trace(duration_s=8.0, seed=4)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    out = runner.run_all(str(tmp_path / "rec.abf"), 0, Settings(),
                         methods=["threshold"])
    out.write()
    with open(out.stem + runner.RUN_SUFFIX, encoding="utf-8") as fh:
        assert json.load(fh)["config_source"] is None


# ------------------------------------------ the commands actually print it

def test_every_command_that_loads_settings_prints_the_source():
    """The real failure mode is a command that works but stays quiet.

    Read as source rather than run: `run` needs a recording and `review` opens
    a window, so executing them here would test the environment instead. What
    must hold is structural -- each handler resolves a source and prints it.
    """
    tree = ast.parse(open(CLI_PY, encoding="utf-8").read())
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}

    for name in ("cmd_run", "cmd_review", "cmd_methods", "cmd_calibrate"):
        body = ast.dump(funcs[name])
        assert "_load_settings" in body or "load_with_source" in body, (
            f"{name} does not resolve a settings source")
        assert "describe" in body, (
            f"{name} loads settings without printing where they came from")

    # cmd_launch is exempt because it delegates: it re-enters cmd_run.
    assert "cmd_run" in ast.dump(funcs["cmd_launch"])


def test_load_settings_returns_both_the_settings_and_the_source():
    """It used to return just the settings. Every caller unpacks two values
    now, and a caller that forgot would fail at runtime, not at import."""
    class Args:
        config = None
        no_filter = False
        tolerance_ms = None
        methods = None

    result = cli._load_settings(Args())
    assert isinstance(result, tuple) and len(result) == 2
    settings, source = result
    assert isinstance(settings, Settings)
    assert isinstance(source, ConfigSource)


def test_methods_prints_the_source_before_the_table(capsys, tmp_path):
    """`methods` is the one command cheap enough to actually run here."""
    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))
    assert cli.main(["methods", "--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "tolerance_ms" in out
    assert out.index("tolerance_ms") < out.index("method "), (
        "the provenance must come before the output it qualifies")


# --------------------------------------------------------------- the GUI

def test_the_parameter_window_says_whether_these_are_the_defaults(tmp_path):
    """A terminal line is easy to miss when a window is open, and a spin box
    showing 1.73 looks exactly like a built-in default."""
    from multimethod.paramwindow import describe_settings_origin

    assert "built-in defaults" in describe_settings_origin(
        Settings(), str(tmp_path / "absent.json"))

    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))
    text = describe_settings_origin(s, cfg)
    assert "differ from the built-in defaults" in text
    assert "cfg.json" in text


def test_the_parameter_window_builds_with_the_banner(tmp_path):
    """The dialog is built, not just the helper called -- a banner referring to
    a name defined after it would pass every other test in this file."""
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets

    from multimethod.paramwindow import edit_settings

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))

    seen = []

    def close_it():
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, QtWidgets.QDialog):
                seen.extend(lbl.text() for lbl in w.findChildren(QtWidgets.QLabel)
                            if "built-in defaults" in lbl.text())
                w.reject()

    QtCore.QTimer.singleShot(200, close_it)
    assert edit_settings(s, "", 0, cfg) is None, "reject means cancelled"
    assert seen, "the dialog carried no label saying where the values came from"
