"""
Doc Scanner Setup
=================
"""
from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "lint": ["black==19.10b0", "isort==4.3.21", "pylint==2.4.4"],
    "test": ["pytest==5.2.2", "pytest-cov==2.8.1", "requests==2.22.0"],
}

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="checkmarx",
    version="0.0.1",
    description="Document OMR",
    license="Proprietary",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires="~=3.7",
    install_requires=requirements,
    tests_require=EXTRAS_REQUIRE["test"],
    extras_require=EXTRAS_REQUIRE,
    zip_safe=True,
    entry_points={"console_scripts": ["checkmarx = checkmarx.main:main"]},
)

