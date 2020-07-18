import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../cgeo")))


# Import the required classes and functions
# pylint: disable=unused-import,wrong-import-position
from cgeo import coco, features, geo, image, other, rs
