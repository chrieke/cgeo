from pathlib import Path
from setuptools import setup, find_packages

parent_dir = Path(__file__).resolve().parent


setup(
    name="cgeo",
    version=parent_dir.joinpath("cgeo/_version.txt").read_text(encoding="utf-8"),
    description="Convenience tools for geospatial processing & remote sensing",
    long_description=parent_dir.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/chrieke/cgeo",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    package_data={
        "": ["_version.txt"]
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=parent_dir.joinpath("requirements.txt").read_text().splitlines()
)