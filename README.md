# mlidk

### Dependencies

```
clang/gcc
llvm-14 // for MacOS: brew install llvm@14
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
./build/mlidk
```

### Test

```
./build/all_tests
```

### Clean

```
cmake --build . --target clean
```

# Test
