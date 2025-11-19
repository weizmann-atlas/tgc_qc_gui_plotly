# TGC QC GUI - Plotly

A PyQt5-based GUI application for viewing and analyzing TGC (Thin Gap Chamber) Quality Control data with interactive Plotly visualizations.

## Features

- **Noise Analysis**: Visualize noise rates per channel per PP half
- **Cosmic Ray Analysis**: Display cosmic ray occupancy heatmaps from ROOT files
- **Threshold Scan**: Analyze threshold scan data across multiple files with error bars
- **Interactive Plots**: All visualizations use Plotly for interactive exploration

## Requirements

- Python 3.6+
- PyQt5
- numpy
- plotly
- uproot (for cosmic ray ROOT file support)

## Installation

```bash
pip install PyQt5 numpy plotly uproot
```

## Usage

```bash
python tgc_qc_gui_plotly.py
```

## Modes

1. **Noise**: Load a single noise text file to visualize noise rates
2. **Cosmic**: Load a ROOT file to display cosmic ray occupancy
3. **Threshold Scan**: Load multiple threshold scan files (e.g., `noise_0mV.txt`, `noise_10mV.txt`, etc.)
4. **Hit Rate**: (TODO - Not yet implemented)

## File Formats

### Noise Files
Text files containing PP blocks with channel data in the format:
```
----PP0
0.123 : 0.456 : 0.789
...
```

### Threshold Scan Files
Multiple text files with naming pattern: `*_<threshold>mV.txt` (e.g., `noise_0mV.txt`, `noise_10mV.txt`)

### Cosmic Files
ROOT files containing a tree with:
- `HitLayer`: Layer number
- `HitIsStrip`: Boolean indicating strip vs wire
- `HitChannel`: Channel number

## PP Channel Mapping

The application uses a predefined mapping of PP halves to layers and channel types:
- PP0A, PP1A, PP4A, PP5A: Layer 0
- PP0B, PP1B, PP4B, PP5B: Layer 1
- PP2A, PP3A, PP6A, PP7A: Layer 2

Use the "Show Mapping" button in the GUI to view the complete mapping.

## License

[Add your license here]

