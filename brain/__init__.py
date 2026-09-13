import sys
from pathlib import Path

# The brain drives the simulator's hands (sim/) and can use the trained policy (train/).
_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "sim"), str(_root / "train")]
