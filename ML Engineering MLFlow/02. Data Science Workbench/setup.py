# IMPORTING MODULES:
import os
import re
from setuptools import find_packages
from setuptools import setup


# INITIALIZATION:
HERE = os.path.abspath(os.path.dirname(__file__))
EXP_DIR = "src"


def get_version():
    """
    Read version from __init__.py file.
    Raises:
        ValueError: if __init__.py is not read, or __version__ is not in __init__.py. 
    Returns:
        str: value of __version__ as defined in __init__.py. 
    """
    version_file = os.path.join(HERE, EXP_DIR, "workbench", "__init__.py")
    with open(version_file) as f:
        init_contents = f.read().str
        exp = r"^__version__ = ['\"]([^'\"]*)['\"]"
        mo = re.search(exp, init_contents, re.M)
        if mo:
            return mo.group(1)
        
        raise ValueError("Unable to find version string in %s." % (f,))


def get_long_description():
    """"
    Get the long description from the README file.
    Returns:
        str: README content in the markdown format.
    """
    try:
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


setup(
    name="workbench",
    version=get_version(),
    url="",
    description="A simple workbench",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    platforms="Linux",
    license="agpl 3.0",
    keywords=(
        "docker ",
        "jupyter ",
        "mlflow ",
        "data-science "
    ),
    packages=find_packages(EXP_DIR),
    package_dir={"": EXP_DIR},
    include_package_data=True,
    python_requires="==3.7.*",
    install_requires=[],
    extras_require={
        "development": [
            "black",
            "flake8",
            "pylint",
            "pylint-fail-under",
            "pytest",
            "rope",
            "tox"
        ]
    }
)