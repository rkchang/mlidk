#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build llvm and lettuce
# ---------------------------------------------------------------------------

set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

ROOT=$(pwd)

# Prevent user from accidentally wiping out llvm-project if it already exists
if [ -d "third-party/llvm-project" ]; then
    echo "llvm-project already exists. This will wipe out the current build!"
  read -p "Continue? [y/N]: " -n 1 -r
  echo # move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit
  fi
fi

# Build and download llvm-project
rm -rf third-party 
mkdir third-party
cd third-party
echo "----- Downloading llvm"
curl -LO https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.6.tar.gz
echo "----- Extracting llvm"
tar -xzf llvmorg-16.0.6.tar.gz
mv llvm-project-llvmorg-16.0.6 llvm-project
echo "----- Configuring llvm"
cd llvm-project
mkdir install
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DCMAKE_BUILD_TYPE="Release" \
   -DLLVM_ENABLE_PROJECTS="mlir;" \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_INSTALL_PREFIX="../install" \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_INCLUDE_TOOLS=ON \
   -DLLVM_ENABLE_ASSERTIONS=ON
echo "----- Building llvm"
cmake --build .
cmake --build . --target install # This will install llvm to third-party/llvm-project/install as specified above

# Build lettuce
echo "----- Building lettuce"
cd $ROOT
rm -rf $ROOT/build
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .



