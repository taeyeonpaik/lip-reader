import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
work = root / 'work'
project = root / 'outputs/lipnet-tutorial'
kernel = work / 'jupyter/kernels/lipnet-tutorial'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python (LipNet TensorFlow)', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(work / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(work / 'jupyter-runtime')
os.environ['IPYTHONDIR'] = str(work / 'ipython')
os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib-cache')
os.environ['LIPNET_DATA'] = str(work / 'tutorial-assets/dataset/data')
notebook = nbformat.read(project / 'LipReading.ipynb', as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='lipnet-tutorial',
                        resources={'metadata': {'path': str(project)}})
client.execute()
nbformat.write(notebook, project / 'LipReading.ipynb')
print('Executed', sum(cell.cell_type == 'code' for cell in notebook.cells), 'code cells')
