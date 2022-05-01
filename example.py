
from model import Model # the Model class wraps the c simulator
import pathlib

pathlib.Path("models").mkdir(parents=True, exist_ok=True) # ensure output location exists

model = Model(
    hypervectorSize=10000, # size of hypervectors used
    inputQuant=2, # # of levels to quantize input image pixels (original is 256 levels)
    classVectorQuant=2, # # of levels to quantize class vectors; the program only
                        # works properly with multiple of 2 values for this;
                        # 0 is a special case that will trigger no quantization
    imageSize=28 # 28x28 is full size image, works with anything down to 9x9
)

model.train(
    trainSamples=60000, # number of train images; 60000 is max # in mnist dataset
    retrainIterations=4, # number of retrain iterations; these are passes through
                         # the training set again that adjusts for classification
                         # errors to try to boost accuracy
    labelsFn=None,  # putting None for these will make it auto select the image
    imagesFn=None   # files for the model's image size
)

nCorrect = model.test(
    testSamples=10000, # number of test images; 10000 is max # in the MNIST dataset
    labelsFn=None,  # putting None for these will make it auto select the image
    imagesFn=None   # files for the model's image size
)

print(f"Test Accuracy: {100 * nCorrect / 10000:.2f}%")

# Example Saving and then loading the same model

model.save("models/example1.hvmodel")
loadedModel = Model.load("models/example1.hvmodel")
nCorrect = loadedModel.test(testSamples=10000)
print(f"Reloaded Model Test Accuracy: {100 * nCorrect / 10000:.2f}%") # should be the same as before
