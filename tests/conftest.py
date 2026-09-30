"""Backend modules import each other by bare name (`import config`), so the
backend directory has to be on sys.path before any test imports one."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
