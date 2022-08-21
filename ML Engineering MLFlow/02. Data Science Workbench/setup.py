# IMPORTING MODULES:
import os
import re
from setuptools import find_packages
from setuptools import setup


# INITIALIZATION:
HERE = os.path.abspath(os.path.dirname(__file__))
EXP_DIR = "src"


def get_version():
    