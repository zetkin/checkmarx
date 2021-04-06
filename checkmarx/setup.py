"""
Doc Scanner Setup
=================
"""
from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "lint": ["black==20.8b1", "isort==5.8.0", "pylint==2.7.4"],
    "test": ["pytest==6.2.3", "pytest-cov==2.11.1"],
}

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="checkmarx",
    version="0.1.0",
    description="Document OMR",
    license="Proprietary",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires="~=3.9",
    install_requires=requirements,
    tests_require=EXTRAS_REQUIRE["test"],
    extras_require=EXTRAS_REQUIRE,
    zip_safe=True,
    entry_points={"console_scripts": ["checkmarx = checkmarx.main:main"]},
)

