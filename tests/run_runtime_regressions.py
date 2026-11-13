r"""Run focused offline regressions with a hard watchdog and D: temp storage.

Windows: .\.venv\Scripts\python.exe tests/run_runtime_regressions.py
"""
from pathlib import Path
import faulthandler
import os
import sys
import tempfile
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    temporary = root / ".bear" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    tempfile.tempdir = str(temporary)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    os.chdir(root)
    faulthandler.dump_traceback_later(60, exit=True)
    try:
        patterns = ["test_session_persistence_regression.py", "test_session_improvements.py",
                    "test_model_request_limits.py", "test_configuration.py",
                    "test_reliability_boundaries.py", "test_runtime_reliability.py",
                    "test_context_preflight.py"]
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(root / "tests"), pattern=p)
                                   for p in patterns)
        return not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
