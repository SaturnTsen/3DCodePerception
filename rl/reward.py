"""verl custom reward entry point. Generated scripts execute in a dedicated Blender process."""
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

import numpy as np
from rl.geometry import compare, sample_surface


def compute_score(data_source, solution_str, ground_truth, extra_info, *, blender, timeout, samples, seed, work_dir):
    with np.load(ground_truth, allow_pickle=False) as mesh:
        target = sample_surface(mesh['vertices'], mesh['faces'], samples, seed)
        scale = np.linalg.norm(np.ptp(mesh['vertices'], axis=0))
    if scale <= 0:
        raise ValueError("target has zero extent")
    blocks = re.findall(r'```python\s*\n(.*?)```', solution_str, re.DOTALL)
    code = blocks[0] if len(blocks) == 1 else solution_str
    failure = {"score": 0.0, "chamfer": -1.0, "execution_success": 0.0}
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as directory:
        root = Path(directory).resolve()
        script, output = root / 'response.py', root / 'mesh.npz'
        script.write_text(code)
        with (root / 'blender.log').open('w') as log:
            process = subprocess.Popen([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', 'import bpy; bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)', '--python', str(script), '--python', str(Path(__file__).with_name('export_mesh.py')), '--', str(output)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=root)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return failure
        if process.returncode != 0 or not output.exists():
            return failure
        try:
            with np.load(output, allow_pickle=False) as mesh:
                prediction = sample_surface(mesh['vertices'], mesh['faces'], samples, seed)
        except ValueError:
            return failure
        return compare(prediction, target, scale)
