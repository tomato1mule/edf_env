from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ros_edf",
    version="0.0.1",
    author="Hyunwoo Ryu",
    author_email="tomato1mule@gmail.com",
    description="ROS interface for Equivariant Descriptor Fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomato1mule/edf_env",
    project_urls={
        "Bug Tracker": "https://github.com/tomato1mule/edf_env/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="<3.9",
    install_requires=[
    ]
)