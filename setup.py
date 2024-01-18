from distutils.core import setup

# --------------------------------------
from enum import IntEnum


class Version(IntEnum):
    major = 0
    minor = 0
    patch = 2


setup(
    name="Pyrception",
    author="Alexander Hadjiivanov",
    version=f"{Version.major}.{Version.minor}.{Version.patch}",
    packages=["pyrception"],
    install_requires=[],
    license="MIT",
    long_description=open("README.md").read(),
)
