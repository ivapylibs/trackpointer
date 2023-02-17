from setuptools import setup, find_packages

setup(
    name="trackpointer",
    version="1.0.1",
    description="Classes implementing object tracking by giving track point or track coordinate frame.",
    author="IVALab",
    packages=find_packages(),
    install_requires=[
        "improcessor @ git+https://github.com/ivapylibs/improcessor.git",
        "detector @ git+https://github.com/ivapylibs/detector.git",
        "Lie @ git+https://github.com/ivapylibs/Lie.git",
    ],
)
