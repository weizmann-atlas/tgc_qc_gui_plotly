import sys
import re
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QSizePolicy, QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QTabWidget, QComboBox, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class TGC_QC_GUI_Plotly(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TGC Noise and Cosmic Viewer - Plotly")
        self.resize(1000, 800)

        self.pp_channel_mapping = {
            'PP0A': {'layer': 0, 'type': 'wire',  'channels': '0–15'},
            'PP1A': {'layer': 0, 'type': 'wire',  'channels': '16–31'},
            'PP2A': {'layer': 2, 'type': 'wire',  'channels': '0–15'},
            'PP3A': {'layer': 2, 'type': 'wire',  'channels': '16–31'},
            'PP4A': {'layer': 0, 'type': 'strip', 'channels': '0–15'},
            'PP5A': {'layer': 0, 'type': 'strip', 'channels': '16–31'},
            'PP0B': {'layer': 1, 'type': 'wire',  'channels': '0–15'},
            'PP1B': {'layer': 1, 'type': 'wire',  'channels': '16–31'},
            'PP4B': {'layer': 1, 'type': 'strip', 'channels': '0–15'},
            'PP5B': {'layer': 1, 'type': 'strip', 'channels': '16–31'},
            'PP6A': {'layer': 2, 'type': 'strip', 'channels': '0–15'},
            'PP7A': {'layer': 2, 'type': 'strip', 'channels': '16–31'}
        }

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("No file loaded")
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Noise", "Cosmic", "Threshold Scan", "Hit Rate (TODO)"])
        self.load_button = QPushButton("Load .txt File")
        self.switch_tab_button = QPushButton("Main Plot")
        self.show_mapping_button = QPushButton("Show Mapping")
        self.update_load_button_label()
        self.mode_selector.currentIndexChanged.connect(self.update_load_button_label)

        self.load_button.clicked.connect(self.load_file)
        self.switch_tab_button.clicked.connect(self.switch_plot_tab)
        self.show_mapping_button.clicked.connect(self.show_mapping_dialog)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.mode_selector)
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.show_mapping_button)
        self.layout.addWidget(self.switch_tab_button)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.layout.addWidget(self.tabs)

    def show_mapping_dialog(self):
        legend_text = "\n".join([
            f"{tag}: L{info['layer']}, {info['type']}, {info['channels']}"
            for tag, info in self.pp_channel_mapping.items()
        ])
        QMessageBox.information(self, "PP Channel Mapping", legend_text)

    def switch_plot_tab(self):
        count = self.tabs.count()
        if count > 0:
            current = self.tabs.currentIndex()
            self.tabs.setCurrentIndex((current + 1) % count)

    def update_load_button_label(self):
        mode = self.mode_selector.currentText()
        if mode == "Threshold Scan":
            self.load_button.setText("Load Threshold Files")
        elif mode == "Cosmic":
            self.load_button.setText("Load Cosmic File")
        elif mode == "Noise":
            self.load_button.setText("Load Noise File")
        else:
            self.load_button.setText("Load File")

    def load_file(self):
        mode = self.mode_selector.currentText()

        if mode == "Threshold Scan":
            file_names, _ = QFileDialog.getOpenFileNames(self, "Open Threshold Files", "", "Text Files (*.txt);;All Files (*)")
            if not file_names:
                return
            self.label.setText(f"Loaded: {len(file_names)} files")
            self.plot_threshold_scan(file_names)
            return

        if mode == "Cosmic":
            file_filter = "ROOT Files (*.root);;All Files (*)"
        else:
            file_filter = "Text Files (*.txt);;All Files (*)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if not file_name:
            return

        self.label.setText(f"Loaded: {file_name}")
        if mode == "Noise":
            with open(file_name, 'r') as f:
                lines = f.readlines()
            self.plot_noise_file(lines)
        elif mode == "Cosmic":
            self.plot_cosmic_root(file_name)
        else:
            html = "<h3>Cosmic mode not implemented yet.</h3>"
            tab = QWebEngineView()
            tab.setHtml(html)
            self.tabs.addTab(tab, f"Plot {self.tabs.count() + 1}")
            self.tabs.setCurrentWidget(tab)

    def parse_pp_blocks(self, lines):
        data = {}
        current_pp = None
        for line in lines:
            line = line.strip()
            if line.startswith("----PP"):
                current_pp = line.strip('-')
                data[current_pp] = []
            elif current_pp and re.match(r"^[0-9.eE+\-]+\s*:\s*[0-9.eE+\-]+\s*:\s*[0-9.eE+\-]+", line):
                parts = [float(x.strip()) for x in line.split(":")]
                if len(parts) == 3:
                    data[current_pp].append(parts)
        return data

    def plot_cosmic_root(self, file_path):
        try:
            import uproot
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", "The 'uproot' module is not installed. Please install it with 'pip install uproot'.")
            return

        with uproot.open(file_path) as f:
            tree = f['tree']
            hit_layer = tree['HitLayer'].array(library='np')
            hit_strip = tree['HitIsStrip'].array(library='np')
            hit_channel = tree['HitChannel'].array(library='np')

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

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Strip Occupancy", "Wire Occupancy"))
        fig.add_trace(go.Heatmap(z=occupancy_strip, x=list(range(32)), y=["L1", "L2", "L3"],
                                 colorscale='Viridis'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=occupancy_wire, x=list(range(32)), y=["L1", "L2", "L3"],
                                 colorscale='Viridis'), row=1, col=2)

        fig.update_layout(title="Cosmic Occupancy Heatmap", height=500)
        html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        tab = QWebEngineView()
        tab.setHtml(html)
        self.tabs.addTab(tab, f"Plot {self.tabs.count() + 1}")
        self.tabs.setCurrentWidget(tab)

    def plot_noise_file(self, lines):
        data = self.parse_pp_blocks(lines)
        pp_names = sorted(data.keys())

        valid_tags = []
        heatmap_data = []

        for pp in pp_names:
            values = data[pp]
            if len(values) != 32:
                continue
            for half, offset in zip(["A", "B"], [0, 16]):
                tag = f"{pp}{half}"
                if tag not in self.pp_channel_mapping:
                    continue
                try:
                    row = [values[ch + offset][1] for ch in range(16)]
                except IndexError:
                    continue
                heatmap_data.append(row)
                valid_tags.append(tag)

        if not heatmap_data:
            return

        hitmap = np.array(heatmap_data).T

        fig = go.Figure(data=go.Heatmap(
            z=hitmap,
            x=valid_tags,
            y=list(range(16)),
            colorscale='Viridis',
            colorbar=dict(title='val1')
        ))

        fig.update_layout(
            title="Noise Rate per Channel per PP Half (val1)",
            xaxis_title="PP Half",
            yaxis_title="Channel (0–15)",
            margin=dict(t=50, b=50)
        )

        html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        tab = QWebEngineView()
        tab.setHtml(html)
        self.tabs.addTab(tab, f"Plot {self.tabs.count() + 1}")
        self.tabs.setCurrentWidget(tab)

    def plot_threshold_scan(self, file_names):
        thresholds = []
        global_data = {'strip': [], 'wire': []}
        layer_data = {0: {'strip': [], 'wire': []}, 1: {'strip': [], 'wire': []}, 2: {'strip': [], 'wire': []}}

        for file in file_names:
            match = re.search(r"_(\d+)mV", file)
            if not match:
                continue
            threshold = int(match.group(1))
            with open(file, 'r') as f:
                data = self.parse_pp_blocks(f.readlines())

            strip_vals, wire_vals = [], []
            layer_vals = {0: {'strip': [], 'wire': []}, 1: {'strip': [], 'wire': []}, 2: {'strip': [], 'wire': []}}

            for pp in data:
                values = data[pp]
                if len(values) != 32:
                    continue
                for half, offset in zip(["A", "B"], [0, 16]):
                    tag = f"{pp}{half}"
                    if tag not in self.pp_channel_mapping:
                        continue
                    info = self.pp_channel_mapping[tag]
                    try:
                        occs = [values[ch + offset][1] for ch in range(16)]
                        layer_vals[info['layer']][info['type']].extend(occs)
                        if info['type'] == 'strip':
                            strip_vals.extend(occs)
                        else:
                            wire_vals.extend(occs)
                    except IndexError:
                        continue

            if not strip_vals or not wire_vals:
                continue

            thresholds.append(threshold)
            global_data['strip'].append((np.mean(strip_vals), np.var(strip_vals)))
            global_data['wire'].append((np.mean(wire_vals), np.var(wire_vals)))

            for lyr in layer_vals:
                for typ in ['strip', 'wire']:
                    vals = layer_vals[lyr][typ]
                    if len(vals):
                        layer_data[lyr][typ].append((np.mean(vals), np.var(vals)))
                    else:
                        layer_data[lyr][typ].append((0, 0))

        fig = make_subplots(rows=2, cols=2, subplot_titles=("Global", "L1", "L2", "L3"))

        for idx, (name, series) in enumerate(global_data.items()):
            means, vars_ = zip(*series)
            fig.add_trace(go.Scatter(x=thresholds, y=means, name=f"{name} (global)",
                                     error_y=dict(type='data', array=vars_, visible=True)), row=1, col=1)

        for lyr in range(3):
            for typ in ['strip', 'wire']:
                means, vars_ = zip(*layer_data[lyr][typ])
                fig.add_trace(go.Scatter(x=thresholds, y=means, name=f"{typ} (L{lyr+1})",
                                         error_y=dict(type='data', array=vars_, visible=True)),
                              row = 1 + (lyr + 1) // 2, col = 1 + (lyr + 1) % 2)

        fig.update_layout(title="Threshold Scan: Avg Occupancy vs Threshold",
                          height=800, margin=dict(t=50, b=50), showlegend=True)

        fig.update_xaxes(title_text="Threshold (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Average Occupancy", row=1, col=1)
        fig.update_xaxes(title_text="Threshold (mV)", row=1, col=2)
        fig.update_yaxes(title_text="Average Occupancy", row=1, col=2)
        fig.update_xaxes(title_text="Threshold (mV)", row=2, col=1)
        fig.update_yaxes(title_text="Average Occupancy", row=2, col=1)
        fig.update_xaxes(title_text="Threshold (mV)", row=2, col=2)
        fig.update_yaxes(title_text="Average Occupancy", row=2, col=2)

        html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        tab = QWebEngineView()
        tab.setHtml(html)
        self.tabs.addTab(tab, f"Plot {self.tabs.count() + 1}")
        self.tabs.setCurrentWidget(tab)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TGC_QC_GUI_Plotly()
    viewer.show()
    sys.exit(app.exec_())
