FROM osrf/ros:jazzy-desktop

RUN apt-get update && apt-get install -y \
    python3-virtualenv \
    git \
    nano \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN virtualenv -p python3 /opt/ros_venv

COPY src/image_bridge/resource/requirements.txt /tmp/requirements.txt

RUN /opt/ros_venv/bin/python -m ensurepip --upgrade || \
    (curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && /opt/ros_venv/bin/python /tmp/get-pip.py)
RUN /opt/ros_venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN /opt/ros_venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /root/project_ws
COPY ./src /root/project_ws/src
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

CMD ["/bin/bash"]
