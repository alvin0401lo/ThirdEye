import shutil
import subprocess
from pathlib import Path

import pytest


def test_microphone_levels_handle_silence_and_signed_full_scale(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed to check firmware microphone statistics")
    sketch = (Path(__file__).resolve().parent.parent / "firmware/thirdeye_ai_device/thirdeye_ai_device.ino").read_text(encoding="utf-8")
    declarations = sketch.split("// BEGIN MICROPHONE LEVELS", 1)[1].split("// END MICROPHONE LEVELS", 1)[0]
    source = "#include <cstdint>\n#include <cassert>\n" + declarations + """
int main() {
  MicrophoneLevels silence;
  silence.add(0);
  assert(silence.samples == 1 && silence.peak == 0 && silence.sumSquares == 0);
  MicrophoneLevels levels;
  levels.add(-32768);
  levels.add(32767);
  levels.add(-3);
  levels.add(4);
  assert(levels.samples == 4 && levels.clipped == 2);
  assert(levels.peak == 32768 && levels.lastValue == 4);
  assert(levels.sumSquares == 2147418138ULL);
  levels = MicrophoneLevels();
  assert(levels.samples == 0 && levels.sumSquares == 0 && levels.peak == 0);
}
"""
    executable = tmp_path / "mic_levels.exe"
    subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-pedantic", "-static", "-x", "c++", "-", "-o", str(executable)], input=source, text=True, capture_output=True, check=True)
    subprocess.run([str(executable)], check=True, timeout=10)
