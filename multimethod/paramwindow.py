"""One window holding every parameter of every method.

Each detector gets a tab, generated from its own params dataclass rather than
hand-written: a new parameter appears here as soon as it exists on the
dataclass. That is the whole reason `settings.field_specs` exists -- a
hand-maintained form would drift from the dataclasses within a release or two,
and a parameter window that silently omits a parameter is worse than none.

Every parameter carries its explanation as a VISIBLE label under the field,
not only as a tooltip. A tooltip has to be hunted for: you have to know it is
there and hover the right widget. The text is the field's own docstring, so
the window cannot drift from the code, and `tests` fails if any parameter
loses its explanation. Each detector tab also opens with what that method is
good at, what it is bad at, and which knob to reach for first.

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


#: Longer per-detector guidance for the parameter window: what each method is
#: good at, what it is bad at, and which knob to reach for first. Kept here
#: rather than in `Detector.blurb` because the blurb also has to fit on one
#: line of the `methods` command and in the README table.
DETECTOR_NOTES = {
    "threshold": (
        "Every accept or reject traces to a parameter you set, which is why "
        "this is the method to reach for when you need to explain a "
        "count.<br>"
        "<b>Weak at:</b> events close together. Only the tallest peak in each "
        "<i>search local max</i> window can be a candidate, so a small event "
        "beside a large one is never considered.<br>"
        "<b>Tune first:</b> <tt>amplitude sd</tt>, then "
        "<tt>search local max ms</tt> \u2014 too short and one event is "
        "reported several times."),
    "deconvolution": (
        "Nothing here needs re-tuning per cell: the kernel comes from the "
        "recording and the thresholds are in units of its own noise.<br>"
        "<b>Strong at:</b> overlapping events, and not fragmenting decay "
        "tails.<br>"
        "<b>Weak at:</b> slow events. Above about 25 ms decay tau nothing is "
        "genuinely isolated, so the estimated kernel becomes unstable \u2014 "
        "the run warns when that happens.<br>"
        "<b>Tune first:</b> <tt>thresh sd</tt>; for slow events raise "
        "<tt>kernel isolation ms</tt>."),
    "template": (
        "The scale and offset are fitted at every position, so only the "
        "template's <i>shape</i> matters \u2014 not its amplitude, and not "
        "the local baseline.<br>"
        "<b>Strong at:</b> small events buried in noise.<br>"
        "<b>Weak at:</b> kinetics unlike the template, and correlated noise "
        "\u2014 the standard error assumes independent samples, which "
        "low-pass filtering breaks.<br>"
        "<b>Tune first:</b> <tt>criterion threshold</tt> (14 here, not the "
        "textbook 3\u20135, because this package filters by default), then "
        "<tt>min sep ms</tt>."),
    "autoclassifier": (
        "The only detector here that uses no labels and no template: it "
        "encodes short windows, clusters them into states, and calls an event "
        "where the trace enters the state that looks like an onset.<br>"
        "<b>Strong at:</b> small events -- it beats every other method here at "
        "SNR 4 on planted data -- and it cannot inherit a labelling bias.<br>"
        "<b>Weak at:</b> knowing when there is nothing to find. Clustering "
        "always returns states, so <tt>min state rise sd</tt> exists to ask "
        "whether any of them actually looks like an event; without it the "
        "method fires on pure noise. Recall is also its lowest score.<br>"
        "<b>Note:</b> an adaptation of Pircher et al. 2022, not a port -- the "
        "encoder here is linear (PCA) rather than an autoencoder, and the rule "
        "naming the onset state is this package's.<br>"
        "<b>Tune first:</b> <tt>n clusters</tt>, then <tt>min sep ms</tt>."),
    "miniml": (
        "A trained network, so it has no thresholds to tune \u2014 but it "
        "also cannot explain a call, and it inherits whatever its training "
        "data contained.<br>"
        "<b>Note:</b> the shipped models were trained on cerebellar granule "
        "cell mEPSCs. On a different preparation it is running "
        "out-of-distribution, which is a good reason to compare it against "
        "the others rather than trust it alone.<br>"
        "<b>Tune first:</b> <tt>threshold</tt>; leave "
        "<tt>window ms</tt> at 12 unless you know the model wants otherwise."),
    "fastmini": (
        "A generic adapter: it runs whatever command you give it and reads "
        "event times out of the CSV that command leaves behind. Only the "
        "times are used \u2014 everything else is measured here, so the "
        "result stays comparable with the other methods.<br>"
        "<b>Set up:</b> tick <tt>enabled</tt>, give a <tt>command</tt>, and "
        "check <tt>time column</tt> matches your CSV."),
}

HELP_STYLE = "color:#6b6b6b; font-size:8pt;"


def _add_fields(QtWidgets, form, obj, entries: dict) -> None:
    """One row per field: the editor, then its explanation underneath.

    The explanation is a visible label rather than only a tooltip. A tooltip
    has to be hunted for -- you have to know it is there and hover the right
    widget -- and these parameters are exactly the ones nobody should have to
    guess at. The text is the field's own docstring, so the window and the
    code cannot drift apart.
    """
    for name, tp, value, doc in field_specs(obj):
        w = _make_widget(QtWidgets, name, tp, value, doc)
        label = QtWidgets.QLabel(name.replace("_", " ") + ":")
        label.setToolTip(doc)
        form.addRow(label, w)
        if doc:
            help_label = QtWidgets.QLabel(doc)
            help_label.setWordWrap(True)
            help_label.setStyleSheet(HELP_STYLE)
            form.addRow("", help_label)
        entries[name] = (w, tp, value)


def describe_settings_origin(settings: Settings, config_path: str | None,
                             ignored: bool = False) -> str:
    """One line for the dialog: are these the built-in defaults or not?

    Mirrors `ConfigSource.describe` for the GUI, where a terminal message is
    easy to miss, but stays short -- the field-by-field diff belongs in the
    terminal, not in a banner.
    """
    from .settings import default_config_path, describe_source

    resolved = config_path or default_config_path()
    exists = os.path.exists(resolved)
    if ignored:
        # Without this branch the banner would compare built-in defaults
        # against built-in defaults, find no difference, and report the config
        # file as merely inert -- when in fact it is being skipped.
        if exists:
            return (f"Built-in defaults (--defaults). "
                    f"{os.path.basename(resolved)} is present but is not being "
                    f"read; \"Save as defaults\" would overwrite it.")
        return "These are the package's built-in defaults."
    src = describe_source(settings, resolved, exists, config_path is not None)
    if not src.overrides:
        if not exists:
            return "These are the package's built-in defaults."
        return (f"Loaded {os.path.basename(resolved)}; every value in it "
                f"matches the built-in defaults.")
    return (f"{len(src.overrides)} value(s) below differ from the built-in "
            f"defaults, loaded from {os.path.basename(resolved)}. "
            f'"Reset to defaults" discards them.')


def edit_settings(settings: Settings | None = None,
                  abf_path: str = "", channel: int = 0,
                  config_path: str | None = None, source=None):
    """Open the parameter window. Returns (settings, abf_path, channel) or None.

    None means cancelled -- the caller must not run anything.

    `source` is the caller's `ConfigSource`, used only for the banner. Passing
    it matters when the caller resolved the settings in a way the window cannot
    infer from `config_path` alone -- `--defaults` being the case in point.
    """
    from PyQt5 import QtCore, QtWidgets

    s = settings or Settings()
    # Keep a live reference: an unassigned QApplication([]) can be collected
    # before the dialog below is constructed.
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Multi-method event detection — Parameters")
    outer = QtWidgets.QVBoxLayout(dialog)

    # -- where these values came from -----------------------------------
    # The fields below are pre-filled from whatever settings the caller
    # resolved, which may be a config file written on some earlier day. Saying
    # so is the whole point: a user who sees "6.00" in a box has no way to know
    # whether that is the package default or a value calibrated last month.
    banner = QtWidgets.QLabel(describe_settings_origin(
        s, config_path, getattr(source, "ignored", False)))
    banner.setWordWrap(True)
    banner.setStyleSheet("QLabel { color: #555; padding: 2px 0 6px 0; }")
    outer.addWidget(banner)

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
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        entries = {}
        _add_fields(QtWidgets, form, obj, entries)
        lay.addLayout(form)
        lay.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, title)
        widgets[key] = entries

    add_dataclass_tab(
        "Preprocessing", s.preprocessing, "__pre__",
        "<b>Applied once, before any detector runs</b>, so every method sees "
        "exactly the same trace \u2014 otherwise their event times would not be "
        "comparable and the consensus table would be meaningless.<br><br>"
        "These settings go into the output filenames, so two differently "
        "preprocessed runs cannot overwrite each other, and the review window "
        "rebuilds the same trace from them.<br><br>"
        "<i>Filtering also moves every SD-based threshold, because it lowers "
        "the noise floor. If you switch it off, expect the detectors to become "
        "more permissive.</i>")
    add_dataclass_tab(
        "Measurement", s.measurement, "__meas__",
        "<b>How an event is quantified once a detector has located it</b> "
        "\u2014 amplitude, rise, decay, area. Shared by every method on "
        "purpose.<br><br>"
        "Each detector has its own idea of what an amplitude is (a peak "
        "average against a fixed window, a model's own baseline, a fitted "
        "scale factor). If each reported its own number, the <tt>amp_*</tt> "
        "columns could not be compared and \u201cmethod A finds smaller "
        "events\u201d might just be two conventions. So detection is "
        "per-method and measurement is shared, and a disagreement in the "
        "output is a disagreement about detection alone.<br><br>"
        "<i>Definitions follow the Mini Analysis conventions, so the numbers "
        "match the sibling minianalysis repo.</i>")

    # -- consensus ------------------------------------------------------
    cons_page = QtWidgets.QWidget()
    cons_lay = QtWidgets.QFormLayout(cons_page)
    cons_note = QtWidgets.QLabel(
        "<b>Two methods count as having found the same event if their peaks "
        "fall within this of each other.</b><br><br>"
        "Detectors put the peak of one event a sample or two apart \u2014 "
        "different smoothing, different snapping \u2014 so an exact match "
        "would report near-total disagreement even where they agree "
        "perfectly.<br><br>"
        "This is a real choice, not a constant: <b>every agreement number in "
        "the output moves with it</b>, which is why it is recorded in the run "
        "sidecar. Widen it and methods look more alike; narrow it and they "
        "look more independent.<br><br>"
        "<i>Two rules keep it honest regardless: clusters cannot chain (each "
        "detection must be within tolerance of the cluster mean), and one "
        "method cannot occupy a cluster twice \u2014 so a method that "
        "fragments one event into three does not look like three methods "
        "agreeing.</i>")
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

        notes = DETECTOR_NOTES.get(name, "")
        head = QtWidgets.QLabel(
            f"<b>{det.label}</b><br>{det.blurb}"
            + (f"<br><br>{notes}" if notes else ""))
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
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        entries = {}
        _add_fields(QtWidgets, form, params, entries)
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
        banner.setText(describe_settings_origin(harvest(), path, ignored=False))
        QtWidgets.QMessageBox.information(
            dialog, "Saved", f"Defaults written to\n{path}\n\n"
            "Every later command picks this file up automatically, including "
            "runs that never mention it. Each run reports it on start and "
            "records it in its _multimethod_run.json, so a result can always "
            "be traced to the parameters that produced it. Rename the file to "
            "go back to the built-in defaults.\n\n"
            "It holds machine-specific paths, so it is gitignored.")

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
