# The Lettuce programming language and compiler/interpreter

Lettuce is a new functional programming language with a simple static type system. This repository contains its corresponding MLIR/LLVM based compiler/interpreter: `lettucec`.

Example lettuce code that checks if a number is even or odd:

```
let
    def isEven(n: i32) -> bool =
        if n == 0 then
            true
        else
            isOdd(n - 1)

    def isOdd(n: i32) -> bool =
        if n == 0 then
            false
        else
            isEven(n - 1)
in
    isEven(31)
```

Demo of compiler mode:

```
~/dev/lettuce/build/lettucec
❯ cat ../../examples/print.lettuce                  # Print out the contents of print.lettuce
let i = 1 in print(4)
~/dev/lettuce/build/lettucec
❯ ./lettucec ../../examples/print.lettuce           # Compile print.lettuce
Wrote output.o
❯ clang++ output.o -o output                        # Link generated object file
❯ ./output                                          # Execute the generated binary `output`
4
```

Demo of interpreter mode:

```
~/dev/lettuce/build/lettucec
❯ ./lettucec                                        # Start interpreter mode
>>> let i = (1 + 2) * 3 in i + 4                    # ((1 + 2) * 3) + 4
Return value: 13
>>> let i = 4 in print(i)                           # Print out the value of variable i
4
Return value: 3
>>> let h = (|x : i32, y : i32| x + y)(1, 2) in h   # Define a function that adds arguments x and y and call it with arguments x=1, y=2
Return value: 3
>>> CTRL-D                                          # End input
```

## Build

Tested on Ubuntu 22.04 and MacOS Ventura 13.4.1 (c)

#### Build prerequisites

```
clang++
cmake
clang-tidy
Ninja/Make
python3
curl
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


## Run

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

## Extra Build Instructions

### Incremental builds

- after running build.sh, you can run the following command to rebuild

```
cd build
cmake --build .
```


### Building Tablegen docs

```
cd build
cmake --build . --target mlir-doc
```

### Docker

Follow the instructions below if you'd prefer to use a docker container to build the project.

```
# Build Docker image
docker build . -t lettuce
# Create container and mount current directory in container
docker run -d --name lettuce -v "$(pwd):/lettuce" lettuce sleep infinity
# Start the container
docker exec -it lettuce bash
# Run build instructions in container
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
