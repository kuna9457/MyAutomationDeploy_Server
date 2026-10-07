"""Backend modules import each other by bare name (`import config`), so the
backend directory has to be on sys.path before any test imports one."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Never download the Upstox MCX master or roll contracts during tests; the
# rollover tests switch it back on explicitly with stubbed data.
os.environ.setdefault("MCX_AUTO_ROLL", "false")
