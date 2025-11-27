# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Share Buyback Strategy Streamlit App - macOS version.

Build with: pyinstaller ShareBuybackApp_mac.spec --clean
Output: dist/ShareBuybackStrategy.app
"""

import os
import sys

# Import packages to get their paths
import streamlit
import plotly
import altair
import pandas
import numpy

# Import PyInstaller utilities for collecting metadata
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

block_cipher = None

# Get paths to package data directories
streamlit_path = os.path.dirname(streamlit.__file__)
plotly_path = os.path.dirname(plotly.__file__)
altair_path = os.path.dirname(altair.__file__)

# Collect data files
datas = [
    # Application files
    ('app.py', '.'),
    ('modules', 'modules'),

    # Streamlit static files and runtime
    (os.path.join(streamlit_path, 'static'), 'streamlit/static'),
    (os.path.join(streamlit_path, 'runtime'), 'streamlit/runtime'),

    # Plotly package data
    (os.path.join(plotly_path, 'package_data'), 'plotly/package_data'),
]

# Add package metadata (required for importlib.metadata)
datas += copy_metadata('streamlit')
datas += copy_metadata('altair')
datas += copy_metadata('pandas')
datas += copy_metadata('numpy')
datas += copy_metadata('plotly')
datas += copy_metadata('pyarrow')
datas += copy_metadata('packaging')
datas += copy_metadata('pydeck')

# Collect streamlit data files
datas += collect_data_files('streamlit')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Streamlit
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.scriptrunner.magic_funcs',
    'streamlit.runtime.caching',
    'streamlit.runtime.state',
    'streamlit.components.v1',
    'streamlit.elements',

    # Plotting
    'plotly',
    'plotly.express',
    'plotly.graph_objects',
    'plotly.subplots',
    'altair',

    # Data processing
    'numpy',
    'pandas',
    'pyarrow',
    'pyarrow.vendored.version',

    # Other dependencies
    'PIL',
    'pkg_resources.py2_warn',
    'cachetools',
    'toml',
    'validators',
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
]

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ShareBuybackStrategy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Mac apps should not show terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .icns file if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ShareBuybackStrategy',
)

app = BUNDLE(
    coll,
    name='ShareBuybackStrategy.app',
    icon=None,  # Add path to .icns file if desired
    bundle_identifier='com.sharebuyback.strategy',
    info_plist={
        'CFBundleShortVersionString': '1.3.0',
        'CFBundleVersion': '1.3.0',
        'NSHighResolutionCapable': True,
    },
)
