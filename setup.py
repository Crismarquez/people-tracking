
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="vtracking",
    version=0.1,
    description="video traking",
    author="Blue Labs",
    author_email="bluelabs.ai@gmail.com",
    python_requires=">=3.8",
    install_requires=[required_packages],
)