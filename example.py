
from model import Model, MNIST_Model # the Model class wraps the c simulator
import pathlib

'''pathlib.Path("models").mkdir(parents=True, exist_ok=True) # ensure output location exists

model = MNIST_Model(
    hypervectorSize=10000, # size of hypervectors used
    inputQuant=2, # # of levels to quantize input image pixels (original is 256 levels)
    classVectorQuant=2, # # of levels to quantize class vectors; the program only
                        # works properly with multiple of 2 values for this;
                        # 0 is a special case that will trigger no quantization
    imageSize=14 # 28x28 is full size image, works with anything down to 9x9
)

print("Training... (this might take a few minutes)")
model.train(
    trainSamples=60000, # number of train images; 60000 is max # in mnist dataset
    retrainIterations=4, # number of retrain iterations; these are passes through
                         # the training set again that adjusts for classification
                         # errors to try to boost accuracy
)

print("Testing...")
nCorrect = model.test(
    testSamples=10000, # number of test images; 10000 is max # in the MNIST dataset
)

print(f"Test Accuracy: {100 * nCorrect / 10000:.2f}%")

# Example Saving and then loading the same model

model.save("models/example1.hvmodel")
loadedModel = MNIST_Model.load("models/example1.hvmodel")
nCorrect = loadedModel.test(testSamples=10000)
print(f"Reloaded Model Test Accuracy: {100 * nCorrect / 10000:.2f}%") # should be the same as before
'''

# Iterative training: do one training iteration at a time and measure accuracy
# on the test set after each one; note this model will have different randomly
# chosen basis vectors so its accuracy might be a bit different

print("\nTraining model with accuracy after each iteration:")
model2 = MNIST_Model(10000, 2, 2, 14)
for i in range(5):
    model2.trainOneIteration(60000)

    nCorrect = model2.test(10000)
    print(f"Accuracy after {i+1} iterations: {100 * nCorrect / 10000:.2f}%")