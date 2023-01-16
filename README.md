# 1. edf_env
Pybullet environment for EDF.
# 2. Installation
## 2.1 Install ROS2 with conda via robostack
https://robostack.github.io/GettingStarted.html
https://github.com/RoboStack/ros-humble
```shell
# if you don't have mamba yet, install it first (not needed when using mambaforge):
conda install mamba -c conda-forge

# now create a new environment
mamba create -n edf_env python=3.9
conda activate edf_env

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channels
conda config --env --add channels robostack
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack-humble

# There is a bug with cryptography==39.0.0, so please downgrade it.
# https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
pip install cryptography==38.0.4

# Install the version of ROS you are interested in:
mamba install ros-humble-desktop  # (or "mamba install ros-noetic-desktop" or "mamba install ros-galactic-desktop")

# optionally, install some compiler packages if you want to e.g. build packages in a colcon_ws:
mamba install compilers cmake pkg-config make ninja colcon-common-extensions

# on Linux and osx (but not Windows) for ROS1 you might want to:
mamba install catkin_tools

# reload environment to activate required scripts before running anything
# on Windows, please restart the Anaconda Prompt / Command Prompt!
conda deactivate
conda activate edf_env

# if you want to use rosdep, also do:
mamba install rosdep
rosdep init  # note: do not use sudo!
rosdep update
```

# 2.2 Install Moveit2
```shell
mamba install ros-humble-moveit
# mamba install colcon-mixin
# mamba install vcstool
```

# 2.3 Install Moveit2 Tutorial
```shell
mamba install colcon-mixin
mamba install vcstool
```