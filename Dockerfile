FROM ubuntu:22.04
CMD bash

# Define build arguments for user ID and group ID, with common defaults
ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_NAME=dev

# Install Ubuntu packages as root.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && \
    apt-get -y install \
    clang \
    clang-format \
    clang-tidy \
    cmake \
    doxygen \
    git \
    g++ \
    python3 \
    ninja-build \
    fish \
    curl \
    sudo \
    ca-certificates \
    --no-install-recommends

# Create a group and user with the specified UID/GID
RUN groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -l -u ${USER_UID} -g ${USER_GID} -m ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME}

# Set the working directory inside the container
WORKDIR /app

# Change ownership of the working directory to the non-root user
# This is important if /app is copied into the container or if files are created here before USER switch
RUN chown ${USER_NAME}:${USER_GID} /app

# Set the user for subsequent commands
USER ${USER_NAME}

# Set environment variable for the user's home directory within the container
ENV HOME=/home/${USER_NAME}
