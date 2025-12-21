FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# python3-pip: for installing python packages
# git: for cloning repositories if needed
# build-essential: for compiling c extensions
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Pre-install common ML libraries to speed up execution and reduce network usage during sandbox runs
# As suggested in the Environment Setup section
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    torch \
    pytest \
    matplotlib \
    seaborn

# Create workspace directory
WORKDIR /workspace

# Copy entrypoint script
# Assumes the build context is the parent directory or docker directory
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set PYTHONPATH to include workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# Define entrypoint
ENTRYPOINT ["/entrypoint.sh"]
