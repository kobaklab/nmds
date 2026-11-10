"""Redirect all results/data paths to a throw-away directory before anything imports paths.py."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="nmds-test-")
os.environ["NMDS_RESULTS"] = os.path.join(_tmp, "results")
os.environ["NMDS_DATA"] = os.path.join(_tmp, "data")
