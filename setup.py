"""
Doc Scanner Setup
=================
"""
from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "lint": ["black==19.10b0", "isort==4.3.21", "pylint==2.4.4"],
    "test": ["pytest==5.2.2", "pytest-cov==2.8.1"],
}

setup(
    name="scanner",
    version="0.0.1",
    description="Document Scanner",
    license="Proprietary",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires="~=3.7",
    install_requires=[
        "fastapi==0.48.0",
        "gunicorn==20.0.4",
        "numpy~=1.16.2",
        "opencv-contrib-python-headless==4.2.0.32",
        "opencv-python-headless==4.2.0.32",
        "pydantic==1.4",
        "python-multipart==0.0.5",
        "pyzbar~=0.1.8",
        "uvicorn==0.11.2",
    ],
    tests_require=EXTRAS_REQUIRE["test"],
    extras_require=EXTRAS_REQUIRE,
    zip_safe=True,
    entry_points={"console_scripts": ["scanner = scanner.main:main"]},
)

