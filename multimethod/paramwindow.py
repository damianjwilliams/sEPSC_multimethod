"""One window holding every parameter of every method.

Each detector gets a tab, generated from its own params dataclass rather than
hand-written: a new parameter appears here as soon as it exists on the
dataclass, with its field docstring as the tooltip. That is the whole reason
`settings.field_specs` exists -- a hand-maintained form would drift from the
dataclasses within a release or two, and a parameter window that silently omits
a parameter is worse than none.

Three tabs come first, because they apply to every method and getting them
wrong invalidates the comparison:

  Preprocessing   filter/resample, applied ONCE for all detectors
  Measurement     how an event is quantified, shared by all detectors
  Consensus       how close two methods must be to count as agreeing

Then one tab per detector, each with an enable checkbox.
"""

from __future__ import annotations

import dataclasses
import os

from . import detectors
from .settings import Settings, field_specs


def _is_bool(tp) -> bool:
    return tp is bool or (isinstance(tp, str) and tp.strip() == "bool")


def _is_int(tp) -> bool:
    return tp is int or (isinstance(tp, str) and tp.strip() == "int")


def _is_float(tp) -> bool:
    return tp is float or (isinstance(tp, str) and tp.strip() == "float")


def _make_widget(QtWidgets, name: str, tp, value, tooltip: str):
    """One editor for one field. Types come from the dataclass annotation.

    Annotations may be strings (`from __future__ import annotations` makes them
    all strings), which is why the checks above accept both forms.
    """
    if name == "direction":
        w = QtWidgets.QComboBox()
        w.addItems(["negative", "positive"])
        w.setCurrentText(str(value))
    elif _is_bool(tp) or isinstance(value, bool):
        w = QtWidgets.QCheckBox()
        w.setChecked(bool(value))
    elif _is_int(tp) or (isinstance(value, int) and not isinstance(value, bool)):
        w = QtWidgets.QSpinBox()
        w.setRange(-10_000_000, 10_000_000)
        w.setValue(int(value))
    elif _is_float(tp) or isinstance(value, float):
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(4)
        w.setRange(-1e9, 1e9)
        w.setSingleStep(0.1)
        w.setValue(float(value))
    else:
        w = QtWidgets.QLineEdit(str(value if value is not None else ""))
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _read_widget(QtWidgets, w, tp, original):
    if isinstance(w, QtWidgets.QComboBox):
        return w.currentText()
    if isinstance(w, QtWidgets.QCheckBox):
        return w.isChecked()
    if isinstance(w, (QtWidgets.QSpinBox,)):
        return int(w.value())
    if isinstance(w, (QtWidgets.QDoubleSpinBox,)):
        return float(w.value())
    text = w.text()
    if isinstance(original, (int, float)) and not isinstance(original, bool):
        try:
            return type(original)(text)
        except ValueError:
            return original
    return text


def edit_settings(settings: Settings | None = None,
                  abf_path: str = "", channel: int = 0,
                  config_path: str | None = None):
    """Open the parameter window. Returns (settings, abf_path, channel) or None.

    None means cancelled -- the caller must not run anything.
    """
    from PyQt5 import QtWidgets

    s = settings or Settings()
    # Keep a live reference: an unassigned QApplication([]) can be collected
    # before the dialog below is constructed.
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Multi-method event detection — Parameters")
    outer = QtWidgets.QVBoxLayout(dialog)

    # -- recording ------------------------------------------------------
    file_box = QtWidgets.QGroupBox("Recording")
    file_form = QtWidgets.QFormLayout(file_box)
    abf_edit = QtWidgets.QLineEdit(abf_path or "")
    browse = QtWidgets.QPushButton("Browse...")

    def on_browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            dialog, "Select recording", abf_edit.text() or "", "Axon ABF (*.abf)")
        if path:
            abf_edit.setText(path)

    browse.clicked.connect(on_browse)
    row = QtWidgets.QHBoxLayout()
    row.addWidget(abf_edit)
    row.addWidget(browse)
    file_form.addRow(".abf file:", row)
    chan = QtWidgets.QSpinBox()
    chan.setRange(0, 15)
    chan.setValue(channel)
    file_form.addRow("Channel:", chan)
    outer.addWidget(file_box)

    tabs = QtWidgets.QTabWidget()
    outer.addWidget(tabs)
    widgets: dict = {}

    def add_dataclass_tab(title: str, obj, key: str, note: str = ""):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        if note:
            lbl = QtWidgets.QLabel(note)
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        form = QtWidgets.QFormLayout()
        entries = {}
        for name, tp, value, doc in field_specs(obj):
            w = _make_widget(QtWidgets, name, tp, value, doc)
            form.addRow(name.replace("_", " ") + ":", w)
            entries[name] = (w, tp, value)
        lay.addLayout(form)
        lay.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, title)
        widgets[key] = entries

    add_dataclass_tab(
        "Preprocessing", s.preprocessing, "__pre__",
        "Applied ONCE, before any detector runs, so every method sees the same "
        "trace. Changing it changes the output filenames, so two differently "
        "preprocessed runs cannot overwrite each other.")
    add_dataclass_tab(
        "Measurement", s.measurement, "__meas__",
        "How an event is quantified once a detector has located it. Shared by "
        "every method on purpose: if each reported its own amplitude, the "
        "amp_* columns would not be comparable across methods and an apparent "
        "difference could just be two conventions.")

    # -- consensus ------------------------------------------------------
    cons_page = QtWidgets.QWidget()
    cons_lay = QtWidgets.QFormLayout(cons_page)
    cons_note = QtWidgets.QLabel(
        "Two methods count as having found the same event if their peaks are "
        "within this. It is a real choice, not a constant: the agreement "
        "numbers move with it, so it is written into the run sidecar.")
    cons_note.setWordWrap(True)
    cons_lay.addRow(cons_note)
    tol = QtWidgets.QDoubleSpinBox()
    tol.setDecimals(2)
    tol.setRange(0.1, 100.0)
    tol.setSingleStep(0.5)
    tol.setValue(float(s.tolerance_ms))
    cons_lay.addRow("tolerance ms:", tol)
    tabs.addTab(cons_page, "Consensus")

    # -- one tab per detector -------------------------------------------
    enable_boxes: dict = {}
    for name in detectors.names():
        det = detectors.get(name)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)

        head = QtWidgets.QLabel(f"<b>{det.label}</b><br>{det.blurb}")
        head.setWordWrap(True)
        lay.addWidget(head)

        box = QtWidgets.QCheckBox("Run this method")
        box.setChecked(s.is_enabled(name))
        enable_boxes[name] = box
        lay.addWidget(box)

        params = s.params_for(name)
        ok, why = det.available(params)
        if det.external:
            status = QtWidgets.QLabel(
                "Runs in a separate interpreter.<br>"
                + ("<span style='color:#2a7f2a'>Available.</span>" if ok
                   else f"<span style='color:#a33'>Not available: {why}</span>"))
            status.setWordWrap(True)
            lay.addWidget(status)

        form = QtWidgets.QFormLayout()
        entries = {}
        for fname, tp, value, doc in field_specs(params):
            w = _make_widget(QtWidgets, fname, tp, value, doc)
            form.addRow(fname.replace("_", " ") + ":", w)
            entries[fname] = (w, tp, value)
        lay.addLayout(form)
        lay.addStretch(1)
        widgets[name] = entries

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, det.name)

    # -- buttons --------------------------------------------------------
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    save_btn = buttons.addButton("Save as defaults",
                                 QtWidgets.QDialogButtonBox.ActionRole)
    reset_btn = buttons.addButton("Reset to defaults",
                                  QtWidgets.QDialogButtonBox.ResetRole)
    outer.addWidget(buttons)
    dialog.setMinimumSize(660, 620)

    def harvest() -> Settings:
        """Read every widget back into a fresh Settings."""
        out = Settings()
        pre_vals = {n: _read_widget(QtWidgets, w, tp, orig)
                    for n, (w, tp, orig) in widgets["__pre__"].items()}
        out.preprocessing = type(s.preprocessing)(**pre_vals)
        meas_vals = {n: _read_widget(QtWidgets, w, tp, orig)
                     for n, (w, tp, orig) in widgets["__meas__"].items()}
        out.measurement = type(s.measurement)(**meas_vals)
        out.tolerance_ms = float(tol.value())
        for name in detectors.names():
            vals = {n: _read_widget(QtWidgets, w, tp, orig)
                    for n, (w, tp, orig) in widgets[name].items()}
            out.set_params_for(name, type(s.params_for(name))(**vals))
            out.enabled[name] = enable_boxes[name].isChecked()
        return out

    def on_save():
        path = harvest().save(config_path)
        QtWidgets.QMessageBox.information(
            dialog, "Saved", f"Defaults written to\n{path}\n\n"
            "This file holds machine-specific paths, so it is gitignored.")

    def on_reset():
        if QtWidgets.QMessageBox.question(
                dialog, "Reset", "Discard every value and reload the built-in "
                "defaults?") != QtWidgets.QMessageBox.Yes:
            return
        dialog.done(2)          # signalled to the caller below

    def on_accept():
        if not abf_edit.text().strip():
            QtWidgets.QMessageBox.warning(dialog, "No recording",
                                          "Choose an .abf file first.")
            return
        if not os.path.isfile(abf_edit.text().strip()):
            QtWidgets.QMessageBox.warning(dialog, "Not found",
                                          "That .abf path does not exist.")
            return
        if not any(b.isChecked() for b in enable_boxes.values()):
            QtWidgets.QMessageBox.warning(dialog, "No methods",
                                          "Enable at least one method.")
            return
        dialog.accept()

    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)
    save_btn.clicked.connect(on_save)
    reset_btn.clicked.connect(on_reset)

    code = dialog.exec_()
    if code == 2:                                    # Reset pressed
        return edit_settings(Settings(), abf_edit.text().strip(),
                             chan.value(), config_path)
    if code != QtWidgets.QDialog.Accepted:
        return None
    return harvest(), abf_edit.text().strip(), chan.value()
