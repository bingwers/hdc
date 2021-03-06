
from model import Model, MNIST_Model, ISOLET_Model # the Model class wraps the c simulator
import pathlib

pathlib.Path("models").mkdir(parents=True, exist_ok=True) # ensure output location exists

model = MNIST_Model(
    hypervectorSize=10000, # size of hypervectors used
    inputQuant=2, # # of levels to quantize input image pixels (original is 256 levels)
    classVectorQuant=2, # # of levels to quantize class vectors; the program only
                        # works properly with multiple of 2 values for this;
                        # 0 is a special case that will trigger no quantization
    imageSize=28 # 28x28 is full size image, works with anything down to 9x9
)

print("MNIST Training... (this might take a few minutes)")
model.train(
    trainSamples=60000, # number of train images; 60000 is max # in mnist dataset
    retrainIterations=4, # number of retrain iterations; these are passes through
                         # the training set again that adjusts for classification
                         # errors to try to boost accuracy
)

print("MNIST Testing...")
nCorrect = model.test(
    testSamples=10000, # number of test images; 10000 is max # in the MNIST dataset
)

print(f"MNIST Test Accuracy: {100 * nCorrect / 10000:.2f}%")

# Measuring latency for encoding and classification
# NOTE: benchmark runs a single thread, whereas train and test run 8 threads, so
# throughput with test will be higher than 1 over the sum of latencies

avgEncodeLatency, avgClassifyLatency = model.benchmark(nTests=1000)
print(f"Average encode latency: {1000*avgEncodeLatency:.4f}ms")
print(f"Average classify latency: {1000*avgClassifyLatency:.4f}ms")

# Example Saving and then loading the same model

model.save("models/example1.hvmodel")
loadedModel = MNIST_Model.load("models/example1.hvmodel")
nCorrect = loadedModel.test(testSamples=10000)
print(f"MNIST Reloaded Model Test Accuracy: {100 * nCorrect / 10000:.2f}%") # should be the same as before

# Iterative training: do one training iteration at a time and measure accuracy
# on the test set after each one; note this model will have different randomly
# chosen basis vectors so its accuracy might be a bit different

print("\nTraining model with accuracy after each iteration:")
model2 = MNIST_Model(10000, 2, 2, 28)
for i in range(10):
    model2.trainOneIteration(60000)

    nCorrect = model2.test(10000)
    print(f"MNIST: Accuracy after {i+1} iterations: {100 * nCorrect / 10000:.2f}%")

# Iterative Training on the ISOLET dataset
print("\tTraining ISOLET model:")
model = ISOLET_Model(10000, 64, 2)
for i in range(10):
    model.trainOneIteration()
    nCorrect = model.test()
    print(f"ISOLET: Accuracy after {i}: {nCorrect/1559*100:.2f}%")
