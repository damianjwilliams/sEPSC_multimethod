"""Tests for `--defaults`: open the window on the values the package ships with.

The parameter window pre-fills from whatever settings the caller resolved, and
that is normally a config file. So with a config in place there was no way to
see the built-in values at all -- the window showed `tolerance_ms` as 0.5
because `calibrate --write` had put it there weeks earlier, and 0.5 looks
exactly like a value that shipped that way.

The subtle part, and the reason most of these tests are about the banner: with
`--defaults` the settings ARE the built-in defaults, so a banner that works by
diffing against `Settings()` finds no difference and cheerfully reports the
config file as merely inert. It is not inert -- it is being skipped, and "Save
as defaults" would overwrite it. That distinction has to be carried in, not
inferred.
"""

from __future__ import annotations

import contextlib
import io
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import cli
from multimethod.settings import Settings, load_with_source


# --------------------------------------------------------- load_with_source

def test_ignore_returns_the_built_in_defaults(tmp_path):
    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))

    loaded, src = load_with_source(cfg)
    assert loaded.tolerance_ms == 0.5, "without the flag, the file wins"

    fresh, src2 = load_with_source(cfg, ignore=True)
    assert fresh.to_dict() == Settings().to_dict()
    assert src2.ignored and src2.exists
    assert src2.overrides == ()


def test_an_ignored_config_is_reported_as_skipped_not_as_absent(tmp_path):
    """It matters which: an absent file cannot be overwritten by Save."""
    cfg = Settings().save(str(tmp_path / "cfg.json"))
    _s, src = load_with_source(cfg, ignore=True)
    text = src.describe()
    assert "--defaults" in text
    assert "not being read" in text
    assert "cfg.json" in text


def test_ignore_with_no_file_present_says_so(tmp_path):
    _s, src = load_with_source(str(tmp_path / "absent.json"), ignore=True)
    assert src.ignored and not src.exists
    assert "built-in defaults" in src.describe()
    assert "not being read" not in src.describe(), (
        "there is nothing to skip, so do not imply a file was skipped")


def test_the_sidecar_records_that_the_config_was_skipped(tmp_path):
    cfg = Settings().save(str(tmp_path / "cfg.json"))
    _s, src = load_with_source(cfg, ignore=True)
    assert json.loads(json.dumps(src.as_dict()))["ignored"] is True


# ------------------------------------------------------------------ the CLI

def _help(cmd) -> str:
    """The real parser's help text, so this tests argparse and not a string."""
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        cmd(["--help"])
    return buf.getvalue()


@pytest.mark.parametrize("name", ["run", "review", "launch", "methods",
                                  "calibrate"])
def test_every_settings_command_offers_the_flag(name):
    text = _help(cli.HANDLERS[name])
    assert "--defaults" in text, f"{name} cannot be asked for the built-in defaults"


def test_load_settings_honours_the_flag():
    class Args:
        config = None
        defaults = True
        no_filter = False
        tolerance_ms = None
        methods = None

    s, source = cli._load_settings(Args())
    assert source.ignored
    assert s.to_dict() == Settings().to_dict()


def test_methods_reports_the_built_in_defaults_with_the_flag(capsys):
    assert cli.main(["methods", "--defaults"]) == 0
    out = capsys.readouterr().out
    assert "built-in defaults" in out
    assert "overrides" not in out.split("method ")[0], (
        "nothing should be reported as overridden when the file is skipped")


# ----------------------------------------------------------------- the GUI

def test_the_banner_does_not_call_a_skipped_config_inert(tmp_path):
    """The regression this flag would otherwise introduce.

    With --defaults the settings equal `Settings()`, so a banner that only
    diffs values would say "every value in it matches the built-in defaults" --
    which reads as "this file is harmless" when the truth is "this file is
    being bypassed".
    """
    from multimethod.paramwindow import describe_settings_origin

    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))

    honest = describe_settings_origin(Settings(), cfg, ignored=True)
    assert "not being read" in honest
    assert "matches the built-in defaults" not in honest

    # and without the flag, the same file is still described as overriding
    assert "differ from the built-in defaults" in describe_settings_origin(s, cfg)


def test_the_window_prefills_the_built_in_values_when_told_to(tmp_path):
    """End to end through the dialog, because the flag is worth nothing if it
    stops at the settings object and never reaches a spin box."""
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets

    from multimethod.paramwindow import edit_settings

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))
    _fresh, src = load_with_source(cfg, ignore=True)

    seen = {}

    def look():
        # No processEvents() here: it reenters the dialog's event loop and the
        # reject() below is lost, which hangs exec_() indefinitely.
        dlgs = [w for w in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(w, QtWidgets.QDialog) and w.isVisible()]
        if dlgs:
            tabs = dlgs[0].findChildren(QtWidgets.QTabWidget)[0]
            for i in range(tabs.count()):
                if tabs.tabText(i) == "Consensus":
                    box = tabs.widget(i).findChildren(QtWidgets.QDoubleSpinBox)[0]
                    seen["tolerance"] = box.value()
            seen["banner"] = next(
                (l.text() for l in dlgs[0].findChildren(QtWidgets.QLabel)
                 if "default" in l.text().lower()), None)
        for d in dlgs:
            d.reject()

    QtCore.QTimer.singleShot(150, look)
    assert edit_settings(Settings(), "", 0, cfg, source=src) is None

    assert seen.get("tolerance") == 3.0, (
        f"the window should show the built-in 3.0 ms, not the config's 0.5; "
        f"got {seen.get('tolerance')}")
    assert "not being read" in (seen.get("banner") or "")


def test_without_the_flag_the_window_prefills_the_config(tmp_path):
    """The contrast case, so the test above cannot pass by the window simply
    ignoring config files altogether."""
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets

    from multimethod.paramwindow import edit_settings

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Settings()
    s.tolerance_ms = 0.5
    cfg = s.save(str(tmp_path / "cfg.json"))
    loaded, src = load_with_source(cfg)

    seen = {}

    def look():
        dlgs = [w for w in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(w, QtWidgets.QDialog) and w.isVisible()]
        if dlgs:
            tabs = dlgs[0].findChildren(QtWidgets.QTabWidget)[0]
            for i in range(tabs.count()):
                if tabs.tabText(i) == "Consensus":
                    box = tabs.widget(i).findChildren(QtWidgets.QDoubleSpinBox)[0]
                    seen["tolerance"] = box.value()
        for d in dlgs:
            d.reject()

    QtCore.QTimer.singleShot(150, look)
    assert edit_settings(loaded, "", 0, cfg, source=src) is None
    assert seen.get("tolerance") == 0.5
