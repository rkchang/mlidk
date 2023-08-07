FROM ubuntu:22.04
CMD bash

# Install Ubuntu packages.
# Please add packages in alphabetical order.
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
    curl
