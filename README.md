# mlidk

### Dependencies

```
clang 16.0.6
llvm 16.0.6 
cmake
clang-tidy
Ninja/Make
```

on MacOS with homebrew: `brew install llvm@16`

### Build

This will download/build llvm and lettuce.
```
chmod +x build.sh
./build.sh
```

#### Tablegen docs
```
cmake --build . --target mlir-doc
```

### Run

```
./lettucec/mlidk <path_to_file>
./lettucec/mlidk examples/hello.mlidk
```

### Test

```
./unittests/all_tests
```

### Clean

```
cmake --build . --target clean
```
