# mlidk

### Dependencies

```
clang/gcc
llvm 16
cmake
clang-tidy
Ninja/Make
```

### Build

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

### Run

```
./build/mlidk <path_to_file>
./build/mlidk examples/hello.mlidk
```

### Test

```
./build/all_tests
```

### Clean

```
cmake --build . --target clean
```
