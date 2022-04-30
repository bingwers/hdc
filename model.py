
# gcc -c -o model.o model.c -O4 -fPIC && gcc -shared -o libmodel.so model.o

import ctypes
import pathlib


class Model:
    lib = ctypes.CDLL(pathlib.Path().absolute() / "libmodel.so")

    def __init__(self, hypervectorSize, inputQuant, classVectorQuant, imageSize):
        if (
            hypervectorSize is None and
            inputQuant is None and
            classVectorQuant is None and
            downsize is None
        ):
            return

        self.lib.Model_new.restype = ctypes.c_void_p
        self.model = self.lib.Model_new(
            ctypes.c_int(hypervectorSize),
            ctypes.c_int(inputQuant),
            ctypes.c_int(classVectorQuant),
            ctypes.c_int(imageSize)
        )
    
    def train(self, labelsFn, imagesFn, trainSamples, retrainIterations):
        self.lib.Model_train(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(imagesFn.encode('utf-8')),
            ctypes.c_int(trainSamples),
            ctypes.c_int(retrainIterations)
        )

    def trainOneIteration(self, labelsFn, imagesFn, trainSamples):
        self.lib.Model_trainOneIteration(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(imagesFn.encode('utf-8')),
            ctypes.c_int(trainSamples)
        )


    def test(self, labelsFn, imagesFn, testSamples):
        self.lib.Model_test.restype = ctypes.c_int
        nCorrect = self.lib.Model_test(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(imagesFn.encode('utf-8')),
            ctypes.c_int(testSamples)
        )

        return int(nCorrect)
    
    def classify(self, image):
        imageArray = (ctypes.c_uint8 * 784)()
        for i in range(784):
            imageArray[i] = image[i]
        
        self.lib.Model_classify.restype = ctypes.c_int
        result = self.lib.Model_classify(self.model, imageArray)

        return int(result)
    
    @staticmethod
    def load(modelFn):
        Model.lib.Model_load.restype = ctypes.c_void_p

        model = Model(None, None, None, None)
        model.model = Model.lib.Model_load(
            ctypes.c_char_p(modelFn.encode('utf-8')),
        )

        return model

    def save(self, modelFn):
        self.lib.Model_save(
            self.model,
            ctypes.c_char_p(modelFn.encode('utf-8'))
        )

    def __del__(self):
        self.lib.Model_delete(self.model)

#model.train("mnist/train-labels.idx1-ubyte", "mnist/train-images.idx3-ubyte", 60000, 0)
#model.save("models/test5.model")
#model.train("mnist/train-labels-28x28-300000-skew.idx1-ubyte", "mnist/train-images-28x28-300000-skew.idx3-ubyte", 300000, 8)
#model.save("models/test7.model")
#nCorrect = model.test("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte", 10000)

#model = Model.load("models/test4.model")
#nCorrect = model.test("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte", 10000)
#print(nCorrect)
