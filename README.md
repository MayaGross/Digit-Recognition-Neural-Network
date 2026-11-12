# Handwritten Digit Recognition – MLP in C++

A feedforward neural network (multilayer perceptron) implemented from scratch in C++ that recognizes handwritten digits from 28×28 grayscale images. No external libraries: the matrix class, the layers and the network are all written by hand, with manual memory management.

The network runs inference with pre-trained weights. Training was not part of the project.

## Example

```
$ ./mlpnetwork parameters/w1 parameters/w2 parameters/w3 parameters/w4 \
               parameters/b1 parameters/b2 parameters/b3 parameters/b4
Please insert image path:
images/im0
        **********
     ***************
     ...
Mlp result: 5 at probability: 0.998
```

The program prints the image as ASCII art (`**` for every pixel above 0.1) and then the predicted digit with its probability. Type `q` to quit.

## Architecture

```
image 28×28 ──vectorize──▶ 784 ──▶ 128 ──▶ 64 ──▶ 20 ──▶ 10 ──argmax──▶ digit
                                 ReLU    ReLU    ReLU   Softmax
```

Each layer computes `activation(W · x + b)`. The index of the largest of the 10 output probabilities is the predicted digit.

## Project structure

| File | Responsibility |
| --- | --- |
| `Matrix.h` / `Matrix.cpp` | A `float` matrix stored in one contiguous row-major array on the heap. Arithmetic operators, element access, `transpose`, `vectorize`, `argmax`, `rref`, binary file reading |
| `Activation.h` / `Activation.cpp` | `relu` and `softmax` in the `activation` namespace |
| `Dense.h` / `Dense.cpp` | One fully connected layer: weights, bias and an activation function pointer |
| `MlpNetwork.h` / `MlpNetwork.cpp` | Four `Dense` layers applied in sequence; returns the digit and its probability |
| `main.cpp` | Loads the weight files, builds the network, reads images and prints results |
| `parameters/` | Pre-trained weights (`w1`–`w4`) and biases (`b1`–`b4`), stored as raw binary `float`s |
| `images/` | Sample images (`im0`–`im9`), 784 binary `float`s each |

## Build and run

Requirements: `g++` and `make` (Linux or WSL).

```
make
./mlpnetwork parameters/w1 parameters/w2 parameters/w3 parameters/w4 parameters/b1 parameters/b2 parameters/b3 parameters/b4
```

Enter image paths relative to the project folder, for example `images/im3`. Note that the file number is not the digit: `im0` is a 5.

## Implementation notes

- **Rule of three.** `Matrix` owns its array, so it implements a deep-copying copy constructor, a copy assignment operator and a destructor that uses `delete[]`.
- **Exception-safe assignment.** `operator=` allocates and copies the new data before releasing the old array, so a failed allocation leaves the object unchanged.
- **Validated construction.** The constructor rejects non-positive dimensions before allocating, so a matrix is never empty and a throwing constructor never leaks.
- **Contiguous storage.** Element `(i, j)` lives at `i * cols + j`. Because of this, `vectorize` only changes the dimensions and runs in O(1).
- **Bounds checking.** `operator()` and `operator[]` throw `std::out_of_range`; mismatched dimensions throw `std::length_error`.
- **Const correctness.** Element access has a `const` overload that returns `float` and a non-`const` overload that returns `float&`.
- **Binary I/O.** Weights and images are read byte-for-byte with `istream::read`, and the file size is checked against the expected matrix size before reading.
- **Decoupled activations.** Each `Dense` layer holds a function pointer, so a layer does not need to know which activation it applies.

Memory was checked with `valgrind --leak-check=full`: no leaks and no invalid accesses.

## Known limitations and possible improvements

- **Softmax overflow.** `softmax` applies `exp` directly to its input, so very large values (for example `[100, 200, 300]`) overflow to `inf` and produce `NaN`. Subtracting the maximum value before `exp` fixes this without changing the result.
- **No move semantics.** Every layer's output is copied into the loop variable. A move constructor and move assignment operator would replace these copies with a pointer swap. Using `std::vector<float>` instead of `new[]` would give correct copy and move behavior for free.
- **Fixed number of layers.** `MlpNetwork` stores its layers in a fixed array, which also forces `Dense` to have a default constructor with no activation. A `std::vector<Dense>` built with `emplace_back` would support any depth and allow that constructor to be removed.
- **Cache-unfriendly multiplication.** Matrix multiplication uses the i-j-k loop order. The i-k-j order reads memory sequentially in the inner loop.
- **Floating-point comparison.** `rref` compares values to exactly `0`; comparing to a small epsilon would be more robust.
