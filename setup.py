import os
import subprocess
import sys

import setuptools

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE_PATH = os.path.join(ROOT_DIR, "src", "fsner", "version.py")

try:

    if "sdist" in sys.argv:
        import semantic_version

        version = subprocess.check_output("git describe --exact-match --tags HEAD", shell=True).decode().strip()
        if version.startswith('v'):
            version = version[1:]
        semantic_version.Version(version)
        with open(VERSION_FILE_PATH, "w") as f:
            f.write(f'__version__ = "{version}"')

    else:
        import re

        print(os.listdir(ROOT_DIR))
        with open(os.path.join(ROOT_DIR, 'PKG-INFO')) as f:
            content = f.read()
            matcher = re.search(r"^Version: (.*)$", content, re.M)
            if matcher:
                version = matcher.group(1).strip()
            else:
                raise RuntimeError("Built version is not matching with PKG-INFO")

            with open(VERSION_FILE_PATH, "w") as f:
                f.write(f'__version__ = "{version}"')

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="fsner",
        version=version,
        author="sayef",
        author_email="hello@sayef.tech",
        description="Few-shot Named Entity Recognition",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/sayef/fsner",
        project_urls={
            "Bug Tracker": "https://github.com/sayef/fsner/issues",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        entry_points={
            'console_scripts': ['fsner=fsner.cmdline:main'],
        },
        python_requires=">=3.6",
        install_requires=["pytorch-lightning==1.5.10", "transformers>=4.16.2"],
        extras_require={
            "dev": [
                "setuptools>=57.4.0",
                "wheel>=0.37.0",
                "semantic-version==2.9.0"
            ]
        }
    )
finally:
    if os.path.exists(VERSION_FILE_PATH):
        os.remove(VERSION_FILE_PATH)
