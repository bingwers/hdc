
# MNIST/ISOLET HDC Simulator

Trains and tests hyperdimensional computing models for MNIST digit classification and ISOLET speech recognition with configurable quantization parameters. The C files in `src` include the HDC implementation, input images are contained in `mnist` directory, and ISOLET inputs are in the `isolete directory. A python wrapper for easy model`training and testing is found in `model.py`.

## Setup
Requires python3, gcc, and a unix terminal. To build the library and run a basic example script run:

```
make all
python3 example.py
```

Comments in `example.py` show how to set up model parameters, train, test, and save/load models.
