# mlidk

### Build

#### Build prerequisites

```
clang
cmake
clang-tidy
Ninja/Make
python3
```

#### Building

```
chmod +x build.sh
./build.sh
```

- This will download/build llvm, lettuce and the unittests.
- `build` will contain the build artifacts for lettucec and unittests.
- `third-party` will be populated with the llvm source code.
- `third-party/llvm-project/install/` will be where llvm is installed

#### Incremental builds

- after running build.sh, you can run the following to rebuild

```
cd build
cmake --build .
```

#### Building Tablegen docs

```
cd build
cmake --build . --target mlir-doc
```

### Run

```
./build/lettucec/lettucec <path_to_file>
./build/lettucec/lettucec examples/hello.lettuce
```

### Test

```
./build/unittests/all_tests
```

### Clean

```
cd build
cmake --build . --target clean
```

## Project structure

### Project layout

```
examples/                   # example lettuce programs
include/                    # headers and tablegen files
include/Romaine/            # IR tablegen files and their headers
lettucec/                   # the lettucec compiler sources
lib/                        # libraries
lib/Romaine                 # IR tablegen files and their sources
third-party/                # contains llvm source and install
unittests/                  # contains googletest unittests
```

- CMake files are in each directory and generally correspond to a project
- For example, `lettucec/CMakeLists.txt` is the cmake file for the `lettucec` compiler and declares
  the `lettucec` executable as well as the flags used to build it.
- You must add new source files to both the `CMakeLists.txt` in `lettucec` and `unittests`

### Tablegen

- each tablegen file corresponds to a header or source file
- For example, `RomaineOps.td` corresponds to `RomaineOps.h`
  - after building, `RomaineOps.h.inc` will be generated in the build directory
  - `RomaineOps.h` includes this file
