import os
import sys
from pathlib import Path


def ensure_pushworld_on_path():
    candidate_paths = []

    env_path = os.getenv("PUSHWORLD_SRC_PATH")
    if env_path:
        candidate_paths.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths.append(repo_root.parent / "pushworld" / "python3" / "src")

    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str

    return None


ensure_pushworld_on_path()
