import sys
import re
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTabWidget, QComboBox, QMessageBox,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox,
    QLineEdit, QFormLayout, QScrollArea, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QStandardPaths
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class TGC_QC_GUI_Plotly(QWidget):
    """GUI application for viewing TGC QC data with Plotly visualizations."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TGC Noise and Cosmic Viewer - Plotly")
        self.resize(1100, 750)
        self.web_profile = QWebEngineProfile.defaultProfile()
        self.web_profile.downloadRequested.connect(self.handle_download_requested)

        self.pp_channel_mapping = {
            'PP1A': {'layer': 0, 'type': 'wire',  'channels': '0–15'},
            'PP2A': {'layer': 0, 'type': 'wire',  'channels': '16–31'},
            'PP3A': {'layer': 2, 'type': 'wire',  'channels': '0–15'},
            'PP4A': {'layer': 2, 'type': 'wire',  'channels': '16–31'},
            'PP5A': {'layer': 0, 'type': 'strip', 'channels': '0–15'},
            'PP6A': {'layer': 0, 'type': 'strip', 'channels': '16–31'},
            'PP1B': {'layer': 1, 'type': 'wire',  'channels': '0–15'},
            'PP2B': {'layer': 1, 'type': 'wire',  'channels': '16–31'},
            'PP3B': {'layer': 2, 'type': 'wire',  'channels': '0–15'},
            'PP4B': {'layer': 2, 'type': 'wire',  'channels': '16–31'},
            'PP5B': {'layer': 1, 'type': 'strip', 'channels': '0–15'},
            'PP6B': {'layer': 1, 'type': 'strip', 'channels': '16–31'},
            'PP7A': {'layer': 2, 'type': 'strip', 'channels': '0–15'},
            'PP7B': {'layer': 2, 'type': 'strip', 'channels': '0–15'},
            'PP8A': {'layer': 2, 'type': 'strip', 'channels': '16–31'},
            'PP8B': {'layer': 2, 'type': 'strip', 'channels': '16–31'}
        }
        self.available_asd_cards = self.sorted_asd_cards(self.pp_channel_mapping.keys())
        self.selected_asd_cards = set(self.available_asd_cards)
        self.thr_asd_data = {}
        self.thr_asd_titles = {}
        self.tab_renderers = {}

        # ── Widgets ────────────────────────────────────────────────────────────
        self.label = QLabel("No file loaded")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setStyleSheet("font-size: 10px; color: gray;")

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Noise", "Cosmic", "Threshold Scan", "Threshold by ASD", "Hit Rate (TODO)"])

        self.load_button = QPushButton("Load .txt File")

        self.log_scale_button = QPushButton("Log Scale: OFF")
        self.log_scale_button.setCheckable(True)

        self.select_asd_button = QPushButton("Select ASD Cards")
        self.asd_selection_label = QLabel("")
        self.asd_selection_label.setWordWrap(True)
        self.asd_selection_label.setStyleSheet("font-size: 10px;")

        self.show_mapping_button = QPushButton("Show Mapping")
        self.save_pdf_button = QPushButton("Save PDF")
        self.save_pdf_button.setVisible(False)
        self.switch_tab_button = QPushButton("Next Tab ▶")

        # ── Signals ────────────────────────────────────────────────────────────
        self.save_pdf_button.clicked.connect(self.save_thr_asd_pdf)
        self.load_button.clicked.connect(self.load_file)
        self.switch_tab_button.clicked.connect(self.switch_plot_tab)
        self.show_mapping_button.clicked.connect(self.show_mapping_dialog)
        self.select_asd_button.clicked.connect(self.show_asd_selection_dialog)
        self.mode_selector.currentIndexChanged.connect(self.update_load_button_label)
        self.mode_selector.currentIndexChanged.connect(self._update_save_pdf_visibility)
        self.log_scale_button.toggled.connect(self.update_log_scale_button_label)
        self.log_scale_button.toggled.connect(self._replot_all_tabs)

        self.update_load_button_label()
        self.update_log_scale_button_label()
        self.update_asd_selection_label()

        # ── Sidebar ────────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 4, 0)
        sidebar_layout.setSpacing(6)

        mode_group = QGroupBox("Mode")
        mg = QVBoxLayout(mode_group)
        mg.addWidget(self.mode_selector)
        mg.addWidget(self.load_button)

        display_group = QGroupBox("Display")
        dg = QVBoxLayout(display_group)
        dg.addWidget(self.log_scale_button)

        asd_group = QGroupBox("ASD Cards")
        ag = QVBoxLayout(asd_group)
        ag.addWidget(self.select_asd_button)
        ag.addWidget(self.asd_selection_label)

        tools_group = QGroupBox("Tools")
        tg = QVBoxLayout(tools_group)
        tg.addWidget(self.show_mapping_button)
        tg.addWidget(self.save_pdf_button)
        tg.addWidget(self.switch_tab_button)

        sidebar_layout.addWidget(mode_group)
        sidebar_layout.addWidget(display_group)
        sidebar_layout.addWidget(asd_group)
        sidebar_layout.addWidget(tools_group)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.label)

        # ── Tabs ───────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._remove_tab)

        # ── Main layout ────────────────────────────────────────────────────────
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        self.setLayout(main_layout)
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.tabs, stretch=1)

    def handle_download_requested(self, download_item):
        """Handle downloads triggered from embedded Plotly controls."""
        suggested_name = download_item.downloadFileName() or "plot.png"
        download_dir = (
            QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
            or os.path.expanduser("~/Downloads")
        )
        default_path = os.path.join(download_dir, suggested_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot As PNG",
            default_path,
            "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            download_item.cancel()
            return

        download_item.setPath(save_path)
        download_item.accept()
        self.label.setText(f"Saving: {save_path}")

    def show_mapping_dialog(self):
        """Display the PP channel mapping information in a dialog."""
        legend_text = "\n".join([
            f"{tag}: L{info['layer']}, {info['type']}, {info['channels']}"
            for tag, info in sorted(self.pp_channel_mapping.items())
        ])
        QMessageBox.information(self, "PP Channel Mapping", legend_text)

    def switch_plot_tab(self):
        """Switch to the next plot tab, cycling back to the first."""
        count = self.tabs.count()
        if count > 0:
            current = self.tabs.currentIndex()
            self.tabs.setCurrentIndex((current + 1) % count)

    def update_load_button_label(self):
        """Update the load button text based on the selected mode."""
        mode = self.mode_selector.currentText()
        if mode in ("Threshold Scan", "Threshold by ASD"):
            self.load_button.setText("Load Threshold Files")
        elif mode == "Cosmic":
            self.load_button.setText("Load Cosmic File")
        elif mode == "Noise":
            self.load_button.setText("Load Noise File")
        else:
            self.load_button.setText("Load File")

    def update_log_scale_button_label(self, checked=None):
        """Update the log scale toggle button label."""
        if checked is None:
            checked = self.log_scale_button.isChecked()
        if checked:
            self.log_scale_button.setText("Log Scale: ON")
        else:
            self.log_scale_button.setText("Log Scale: OFF")

    def sorted_asd_cards(self, cards):
        """Return ASD card tags sorted as PP1A, PP1B, PP2A, ..."""
        def sort_key(tag):
            match = re.fullmatch(r"PP(\d+)([A-Z])", tag)
            if match:
                return int(match.group(1)), match.group(2)
            return float("inf"), tag
        return sorted(cards, key=sort_key)

    def update_asd_selection_label(self):
        """Update the summary label for selected ASD cards."""
        total = len(self.available_asd_cards)
        selected = self.sorted_asd_cards(self.selected_asd_cards)
        selected_count = len(selected)

        if selected_count == 0:
            summary = "ASD cards: none selected"
        elif selected_count == total:
            summary = f"ASD cards: all ({total})"
        elif selected_count <= 6:
            summary = "ASD cards: " + ", ".join(selected)
        else:
            summary = f"ASD cards: {selected_count}/{total} selected"
        self.asd_selection_label.setText(summary)

    def _update_save_pdf_visibility(self):
        self.save_pdf_button.setVisible(
            self.mode_selector.currentText() == "Threshold by ASD"
        )

    def show_asd_selection_dialog(self):
        """Allow the user to choose which ASD cards to include in threshold scans."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select ASD Cards")
        dialog_layout = QVBoxLayout(dialog)

        info = QLabel("Choose PPxA/PPxB cards to include in threshold scan calculations:")
        dialog_layout.addWidget(info)

        list_widget = QListWidget(dialog)
        for card in self.available_asd_cards:
            item = QListWidgetItem(card)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if card in self.selected_asd_cards else Qt.Unchecked)
            list_widget.addItem(item)
        dialog_layout.addWidget(list_widget)

        quick_actions_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        clear_all_button = QPushButton("Clear All")
        quick_actions_layout.addWidget(select_all_button)
        quick_actions_layout.addWidget(clear_all_button)
        dialog_layout.addLayout(quick_actions_layout)

        def set_all(checked):
            state = Qt.Checked if checked else Qt.Unchecked
            for idx in range(list_widget.count()):
                list_widget.item(idx).setCheckState(state)

        select_all_button.clicked.connect(lambda: set_all(True))
        clear_all_button.clicked.connect(lambda: set_all(False))

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        selected_cards = {
            list_widget.item(idx).text()
            for idx in range(list_widget.count())
            if list_widget.item(idx).checkState() == Qt.Checked
        }
        if not selected_cards:
            QMessageBox.warning(self, "Warning", "Select at least one ASD card.")
            return

        self.selected_asd_cards = selected_cards
        self.update_asd_selection_label()

    def _ask_asd_titles(self, tags):
        """Show a dialog to set a custom title for each ASD card. Returns {tag: title} or None on cancel."""
        dialog = QDialog(self)
        dialog.setWindowTitle("ASD Card Titles")
        outer_layout = QVBoxLayout(dialog)
        outer_layout.addWidget(QLabel("Enter a title for each ASD card (used in charts and PDF):"))

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        edits = {}
        for tag in tags:
            edit = QLineEdit(tag)
            form_layout.addRow(tag, edit)
            edits[tag] = edit

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setMinimumHeight(min(300, 40 * len(tags) + 20))
        outer_layout.addWidget(scroll)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        outer_layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return {tag: edits[tag].text().strip() or tag for tag in tags}

    def _add_renderable_tab(self, render_func, label):
        view = QWebEngineView()
        view.setHtml(render_func())
        self.tab_renderers[view] = render_func
        self.tabs.addTab(view, label)
        return view

    def _remove_tab(self, idx):
        widget = self.tabs.widget(idx)
        self.tab_renderers.pop(widget, None)
        self.tabs.removeTab(idx)

    def _replot_all_tabs(self):
        for view, render_func in self.tab_renderers.items():
            view.setHtml(render_func())

    def load_file(self):
        """Load and process files based on the selected mode."""
        mode = self.mode_selector.currentText()

        if mode in ("Threshold Scan", "Threshold by ASD"):
            file_names, _ = QFileDialog.getOpenFileNames(
                self, "Open Threshold Files", "", "Text Files (*.txt);;All Files (*)"
            )
            if not file_names:
                return
            self.label.setText(f"Loaded: {len(file_names)} files")
            if mode == "Threshold Scan":
                self.plot_threshold_scan(file_names)
            else:
                self.plot_threshold_scan_by_asd(file_names)
            return

        if mode == "Cosmic":
            file_filter = "ROOT Files (*.root);;All Files (*)"
        else:
            file_filter = "Text Files (*.txt);;All Files (*)"

        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if not file_name:
            return

        self.label.setText(f"Loaded: {file_name}")

        try:
            if mode == "Noise":
                with open(file_name, 'r') as f:
                    lines = f.readlines()
                self.plot_noise_file(lines)
            elif mode == "Cosmic":
                self.plot_cosmic_root(file_name)
            else:
                html = "<h3>Mode not implemented yet.</h3>"
                tab = QWebEngineView()
                tab.setHtml(html)
                self.tabs.addTab(tab, f"Plot {self.tabs.count() + 1}")
                self.tabs.setCurrentWidget(tab)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def parse_pp_blocks(self, lines):
        """Parse PP blocks from text file lines.

        Supports both legacy files (`PPn` blocks with 32 channels) and
        newer files (`PPnA`/`PPnB` blocks with 16 channels each).
        """
        data = {}
        current_pp = None
        incomplete_pps = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, header, and footer lines
            if (not line or 
                line.startswith("data checked") or 
                re.match(r"^\d+$", line) or  # Footer number (e.g., "118")
                re.match(r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat) ", line)):  # Date footer
                continue
            
            if line.startswith("----PP"):
                # Keep half tags when present (e.g. PP1A, PP1B).
                match = re.search(r"PP\d+[A-Z]?", line)
                if match:
                    current_pp = match.group(0)
                    data[current_pp] = []
            elif current_pp:
                match = re.match(
                    r"^([0-9.eE+\-]+)\s*:\s*([0-9.eE+\-]+)\s*:\s*([0-9.eE+\-]+)\s*$",
                    line
                )
                if not match:
                    continue
                try:
                    parts = [float(match.group(i)) for i in range(1, 4)]
                    if not any(np.isnan(parts)) and not any(np.isinf(parts)):
                        data[current_pp].append(parts)
                except (ValueError, IndexError):
                    continue
        
        # Validate PP block completeness
        for pp, values in data.items():
            expected_channels = 16 if re.fullmatch(r"PP\d+[A-Z]", pp) else 32
            if len(values) != expected_channels:
                incomplete_pps.append(f"{pp} ({len(values)}/{expected_channels} channels)")
        
        if incomplete_pps:
            QMessageBox.warning(
                self, "Incomplete Data",
                f"Some PP blocks have incomplete data:\n" + "\n".join(incomplete_pps[:10]) +
                (f"\n... and {len(incomplete_pps) - 10} more" if len(incomplete_pps) > 10 else "")
            )
        
        return data

    def iter_pp_half_blocks(self, data):
        """Yield `(tag, 16-channel values)` for all PP halves in parsed data."""
        for pp in sorted(data.keys()):
            values = data[pp]

            # New format: PPnA / PPnB already split in 16-channel blocks.
            if re.fullmatch(r"PP\d+[A-Z]", pp):
                if pp in self.pp_channel_mapping and len(values) >= 16:
                    yield pp, values[:16]
                continue

            # Legacy format: PPn with 32 channels, split into A/B halves.
            if re.fullmatch(r"PP\d+", pp) and len(values) >= 32:
                for half, offset in (("A", 0), ("B", 16)):
                    tag = f"{pp}{half}"
                    if tag in self.pp_channel_mapping:
                        yield tag, values[offset:offset + 16]

    def plot_cosmic_root(self, file_path):
        """Plot cosmic ray occupancy from a ROOT file."""
        try:
            import uproot
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "The 'uproot' module is not installed. Please install it with 'pip install uproot'."
            )
            return

        try:
            with uproot.open(file_path) as f:
                tree = f['tree']
                hit_layer = tree['HitLayer'].array(library='np')
                hit_strip = tree['HitIsStrip'].array(library='np')
                hit_channel = tree['HitChannel'].array(library='np')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read ROOT file: {str(e)}")
            return

        occupancy_strip = np.zeros((3, 32))
        occupancy_wire = np.zeros((3, 32))

        for layers, strips, channels in zip(hit_layer, hit_strip, hit_channel):
            for layer, is_strip, channel in zip(layers, strips, channels):
                if channel < 0 or channel >= 32:
                    continue
                if is_strip:
                    occupancy_strip[layer][channel] += 1
                else:
                    occupancy_wire[layer][channel] += 1

        def render(occ_strip=occupancy_strip.copy(), occ_wire=occupancy_wire.copy()):
            use_log = self.log_scale_button.isChecked()
            colorbar_title = "count"
            s, w = occ_strip.copy(), occ_wire.copy()
            if use_log:
                for arr in (s, w):
                    pos = arr > 0
                    min_pos = float(np.min(arr[pos])) if np.any(pos) else 1.0
                    arr[:] = np.log10(np.where(pos, arr, min_pos))
                colorbar_title = "log10(count)"
            title_suffix = " (log z)" if use_log else ""
            f = make_subplots(rows=1, cols=2,
                              subplot_titles=("Strip Occupancy", "Wire Occupancy"))
            f.add_trace(go.Heatmap(z=s, x=list(range(32)), y=["L0", "L1", "L2"],
                                   colorscale='Viridis',
                                   colorbar=dict(title=colorbar_title)), row=1, col=1)
            f.add_trace(go.Heatmap(z=w, x=list(range(32)), y=["L0", "L1", "L2"],
                                   colorscale='Viridis', showscale=False), row=1, col=2)
            f.update_layout(title=f"Cosmic Occupancy Heatmap{title_suffix}", height=500)
            return pio.to_html(f, full_html=True, include_plotlyjs='cdn')

        tab_label = f"Plot {self.tabs.count() + 1}"
        view = self._add_renderable_tab(render, tab_label)
        self.tabs.setCurrentWidget(view)

    def plot_noise_file(self, lines):
        """Plot noise data from parsed text file lines."""
        data = self.parse_pp_blocks(lines)
        valid_tags = []
        heatmap_data = []

        for tag, values in self.iter_pp_half_blocks(data):
            row = [channel_vals[0] for channel_vals in values]
            if len(row) != 16:
                continue
            heatmap_data.append(row)
            valid_tags.append(tag)

        if not heatmap_data:
            QMessageBox.warning(self, "Warning", "No valid data found in file.")
            return

        hitmap = np.array(heatmap_data).T

        def render(hm=hitmap, tags=valid_tags):
            use_log = self.log_scale_button.isChecked()
            z_title = "val1"
            if use_log:
                positive_mask = hm > 0
                if not np.any(positive_mask):
                    return "<h3>Log scale: no positive values in data.</h3>"
                min_positive = float(np.min(hm[positive_mask]))
                # Plotly heatmap colorbars do not support native log scaling.
                # Transform z explicitly and clamp non-positive values to the lowest positive entry.
                z_data = np.log10(np.where(positive_mask, hm, min_positive))
                z_title = "log10(val1)"
            else:
                z_data = hm
            title_suffix = " (log z)" if use_log else ""
            fig = go.Figure(data=go.Heatmap(
                z=z_data, x=tags, y=list(range(16)),
                colorscale='Viridis', colorbar=dict(title=z_title)
            ))
            fig.update_layout(
                title=f"Noise Rate per Channel per PP Half (val1){title_suffix}",
                xaxis_title="PP Half", yaxis_title="Channel (0–15)",
                margin=dict(t=50, b=50)
            )
            return pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

        tab_label = f"Plot {self.tabs.count() + 1}"
        view = self._add_renderable_tab(render, tab_label)
        self.tabs.setCurrentWidget(view)

    def plot_threshold_scan_by_asd(self, file_names):
        """Plot mean val1 vs threshold for each selected ASD card individually."""
        selected_asd_cards = set(self.selected_asd_cards)
        if not selected_asd_cards:
            QMessageBox.warning(self, "Warning", "No ASD cards selected.")
            return

        per_card_data = {tag: [] for tag in selected_asd_cards}

        for file_path in file_names:
            match = re.search(r"_(\d+)mV", file_path)
            if not match:
                continue
            threshold = int(match.group(1))

            try:
                with open(file_path, 'r') as f:
                    data = self.parse_pp_blocks(f.readlines())
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to read file {file_path}: {str(e)}")
                continue

            for tag, values in self.iter_pp_half_blocks(data):
                if tag not in selected_asd_cards:
                    continue
                occs = [ch[0] for ch in values]
                per_card_data[tag].append((threshold, np.mean(occs), np.std(occs)))

        ordered_tags = self.sorted_asd_cards(
            tag for tag, pts in per_card_data.items() if pts
        )
        if not ordered_tags:
            QMessageBox.warning(self, "Warning", "No valid threshold data found.")
            return

        titles = self._ask_asd_titles(ordered_tags)
        if titles is None:
            return

        for tag in ordered_tags:
            per_card_data[tag].sort(key=lambda t: t[0])

        self.thr_asd_data = {
            tag: {
                'thresholds': [pt[0] for pt in per_card_data[tag]],
                'means':      [pt[1] for pt in per_card_data[tag]],
                'stds':       [pt[2] for pt in per_card_data[tag]],
            }
            for tag in ordered_tags
        }
        self.thr_asd_titles = titles

        first_new_index = self.tabs.count()

        for tag in ordered_tags:
            d = self.thr_asd_data[tag]
            title = titles[tag]
            info = self.pp_channel_mapping.get(tag, {})
            subtitle = (
                f"Layer {info.get('layer', '?')}, "
                f"{info.get('type', '?')}, "
                f"ch {info.get('channels', '?')}"
            )

            def render(d=d, title=title, tag=tag, subtitle=subtitle):
                use_log = self.log_scale_button.isChecked()
                if use_log:
                    means_plot = [m if m > 0 else None for m in d['means']]
                    y_axis_title = "Mean val1 [log]"
                else:
                    means_plot = d['means']
                    y_axis_title = "Mean val1"
                fig = go.Figure(data=go.Scatter(
                    x=d['thresholds'], y=means_plot,
                    mode='lines+markers',
                    error_y=dict(type='data', array=d['stds'], visible=True),
                    marker=dict(color='steelblue')
                ))
                fig.update_layout(
                    title=f"{title}<br><sup>{tag} — {subtitle}</sup>",
                    xaxis_title="Threshold (mV)", yaxis_title=y_axis_title,
                    margin=dict(t=70, b=50)
                )
                if use_log:
                    fig.update_yaxes(type='log')
                return pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

            self._add_renderable_tab(render, title)

        self.tabs.setCurrentIndex(first_new_index)

    def save_thr_asd_pdf(self):
        """Export all per-ASD threshold scan charts to a single multi-page PDF."""
        if not self.thr_asd_data:
            QMessageBox.warning(self, "Warning", "No Threshold by ASD data loaded. Load files first.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "The 'matplotlib' module is not installed. Please install it with 'pip install matplotlib'."
            )
            return

        download_dir = (
            QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
            or os.path.expanduser("~/Downloads")
        )
        default_path = os.path.join(download_dir, "threshold_by_asd.pdf")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Threshold by ASD PDF", default_path, "PDF Files (*.pdf);;All Files (*)"
        )
        if not save_path:
            return

        use_log = self.log_scale_button.isChecked()

        try:
            with PdfPages(save_path) as pdf:
                for tag in self.sorted_asd_cards(self.thr_asd_data.keys()):
                    d = self.thr_asd_data[tag]
                    title = self.thr_asd_titles.get(tag, tag)
                    info = self.pp_channel_mapping.get(tag, {})
                    subtitle = (
                        f"Layer {info.get('layer', '?')}, "
                        f"{info.get('type', '?')}, "
                        f"ch {info.get('channels', '?')}"
                    )

                    thresholds = d['thresholds']
                    means = np.array(d['means'], dtype=float)
                    stds = np.array(d['stds'], dtype=float)

                    if use_log:
                        valid = means > 0
                        means_plot = np.where(valid, means, np.nan)
                        stds_plot = np.where(valid, stds, np.nan)
                        hidden = int(np.sum(~valid))
                        y_label = "Mean val1 [log]"
                    else:
                        means_plot = means
                        stds_plot = stds
                        hidden = 0
                        y_label = "Mean val1"

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.errorbar(thresholds, means_plot, yerr=stds_plot,
                                marker='o', capsize=4, color='steelblue')
                    if use_log:
                        ax.set_yscale('log')
                    page_title = f"{title}\n{tag} — {subtitle}"
                    if hidden:
                        page_title += f"\n({hidden} non-positive point(s) hidden in log scale)"
                    ax.set_title(page_title, fontsize=11)
                    ax.set_xlabel("Threshold (mV)", fontsize=10)
                    ax.set_ylabel(y_label, fontsize=10)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            self.label.setText(f"PDF saved: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save PDF: {str(e)}")

    def plot_threshold_scan(self, file_names):
        """Plot threshold scan data from multiple files."""
        selected_asd_cards = set(self.selected_asd_cards)
        if not selected_asd_cards:
            QMessageBox.warning(self, "Warning", "No ASD cards selected for threshold scan.")
            return

        thresholds = []
        global_data = {'strip': [], 'wire': []}
        layer_data = {
            0: {'strip': [], 'wire': []},
            1: {'strip': [], 'wire': []},
            2: {'strip': [], 'wire': []}
        }

        for file_path in file_names:
            match = re.search(r"_(\d+)mV", file_path)
            if not match:
                continue
            threshold = int(match.group(1))
            
            try:
                with open(file_path, 'r') as f:
                    data = self.parse_pp_blocks(f.readlines())
            except Exception as e:
                QMessageBox.warning(
                    self, "Warning",
                    f"Failed to read file {file_path}: {str(e)}"
                )
                continue

            strip_vals, wire_vals = [], []
            layer_vals = {
                0: {'strip': [], 'wire': []},
                1: {'strip': [], 'wire': []},
                2: {'strip': [], 'wire': []}
            }

            for tag, values in self.iter_pp_half_blocks(data):
                if tag not in selected_asd_cards:
                    continue
                info = self.pp_channel_mapping[tag]
                occs = [channel_vals[0] for channel_vals in values]
                layer_vals[info['layer']][info['type']].extend(occs)
                if info['type'] == 'strip':
                    strip_vals.extend(occs)
                else:
                    wire_vals.extend(occs)

            if not strip_vals and not wire_vals:
                continue

            thresholds.append(threshold)
            # Use standard deviation instead of variance for error bars
            if strip_vals:
                global_data['strip'].append((np.mean(strip_vals), np.std(strip_vals)))
            else:
                global_data['strip'].append((np.nan, 0))
            if wire_vals:
                global_data['wire'].append((np.mean(wire_vals), np.std(wire_vals)))
            else:
                global_data['wire'].append((np.nan, 0))

            for lyr in layer_vals:
                for typ in ['strip', 'wire']:
                    vals = layer_vals[lyr][typ]
                    if len(vals):
                        layer_data[lyr][typ].append((np.mean(vals), np.std(vals)))
                    else:
                        layer_data[lyr][typ].append((np.nan, 0))

        if not thresholds:
            QMessageBox.warning(self, "Warning", "No valid threshold data found.")
            return

        # Sort by threshold value
        sorted_indices = np.argsort(thresholds)
        thresholds = [thresholds[i] for i in sorted_indices]
        
        for key in global_data:
            global_data[key] = [global_data[key][i] for i in sorted_indices]
        for lyr in layer_data:
            for typ in ['strip', 'wire']:
                layer_data[lyr][typ] = [layer_data[lyr][typ][i] for i in sorted_indices]

        selected_sorted = self.sorted_asd_cards(selected_asd_cards)
        cards_by_type = {'strip': [], 'wire': []}
        cards_by_layer_type = {
            0: {'strip': [], 'wire': []},
            1: {'strip': [], 'wire': []},
            2: {'strip': [], 'wire': []}
        }
        for card in selected_sorted:
            info = self.pp_channel_mapping.get(card)
            if not info:
                continue
            cards_by_type[info['type']].append(card)
            cards_by_layer_type[info['layer']][info['type']].append(card)

        def format_card_list(cards):
            return ", ".join(cards) if cards else "none"

        # Build title string (fixed regardless of log scale)
        if len(selected_sorted) == len(self.available_asd_cards):
            selected_title = "all ASD cards"
        elif len(selected_sorted) <= 4:
            selected_title = ", ".join(selected_sorted)
        else:
            selected_title = f"{len(selected_sorted)} ASD cards"

        def render(thr=thresholds, gd=global_data, ld=layer_data,
                   cbt=cards_by_type, cblt=cards_by_layer_type, st=selected_title):
            use_log = self.log_scale_button.isChecked()
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Global", "L1", "L2", "L3"))
            trace_count = 0

            for name, series in gd.items():
                means, stds = zip(*series)
                means_plot = [m if m > 0 else np.nan for m in means] if use_log else list(means)
                if not np.any(~np.isnan(np.array(means_plot, dtype=float))):
                    continue
                fig.add_trace(go.Scatter(
                    x=thr, y=means_plot,
                    name=f"{name} (global): {format_card_list(cbt[name])}",
                    error_y=dict(type='data', array=list(stds), visible=True)
                ), row=1, col=1)
                trace_count += 1

            for lyr in range(3):
                for typ in ['strip', 'wire']:
                    means, stds = zip(*ld[lyr][typ])
                    means_plot = [m if m > 0 else np.nan for m in means] if use_log else list(means)
                    if not np.any(~np.isnan(np.array(means_plot, dtype=float))):
                        continue
                    # Map layers to subplot positions: L1->(1,2), L2->(2,1), L3->(2,2)
                    row = 1 + (lyr + 1) // 2
                    col = 1 + (lyr + 1) % 2
                    fig.add_trace(go.Scatter(
                        x=thr, y=means_plot,
                        name=f"{typ} (L{lyr+1}): {format_card_list(cblt[lyr][typ])}",
                        error_y=dict(type='data', array=list(stds), visible=True)
                    ), row=row, col=col)
                    trace_count += 1

            if trace_count == 0:
                return "<h3>No plottable data for the selected ASD cards.</h3>"

            y_suffix = " [log]" if use_log else ""
            fig.update_layout(
                title=f"Threshold Scan: Avg Occupancy vs Threshold{y_suffix} ({st})",
                height=800, margin=dict(t=50, b=50), showlegend=True
            )
            for r in [1, 2]:
                for c in [1, 2]:
                    fig.update_xaxes(title_text="Threshold (mV)", row=r, col=c)
                    fig.update_yaxes(title_text="Average Occupancy", row=r, col=c)
                    if use_log:
                        fig.update_yaxes(type='log', row=r, col=c)
            return pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

        tab_label = f"Plot {self.tabs.count() + 1}"
        view = self._add_renderable_tab(render, tab_label)
        self.tabs.setCurrentWidget(view)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TGC_QC_GUI_Plotly()
    viewer.show()
    sys.exit(app.exec_())
