
# gcc -c -o model.o model.c -O4 -fPIC && gcc -shared -o libmodel.so model.o

import ctypes
import pathlib
import math
import os

class Model:
    lib = ctypes.CDLL(pathlib.Path().absolute() / "bin" / "libmodel.so")

    def __init__(self, hypervectorSize, inputQuant, classVectorQuant, featureSize, nClasses):
        if (
            hypervectorSize is None and
            inputQuant is None and
            classVectorQuant is None and
            featureSize is None and
            nClasses is None
        ):
            return

        self.featureSize = featureSize
        self.lib.Model_new.restype = ctypes.c_void_p
        self.model = self.lib.Model_new(
            ctypes.c_int(hypervectorSize),
            ctypes.c_int(inputQuant),
            ctypes.c_int(classVectorQuant),
            ctypes.c_int(featureSize),
            ctypes.c_int(nClasses)
        )
    
    def train(self, trainSamples, retrainIterations, labelsFn, featuresFn):
        self.lib.Model_train(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(featuresFn.encode('utf-8')),
            ctypes.c_int(trainSamples),
            ctypes.c_int(retrainIterations)
        )

    def trainOneIteration(self, trainSamples, labelsFn, featuresFn):
        self.lib.Model_trainOneIteration(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(featuresFn.encode('utf-8')),
            ctypes.c_int(trainSamples)
        )

    def test(self, testSamples, labelsFn, featuresFn):

        self.lib.Model_test.restype = ctypes.c_int
        nCorrect = self.lib.Model_test(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(featuresFn.encode('utf-8')),
            ctypes.c_int(testSamples)
        )

        return int(nCorrect)
    
    def classify(self, features):
        featureArray = (ctypes.c_uint8 * 784)()
        for i in range(784):
            featureArray[i] = features[i]
        
        self.lib.Model_classify.restype = ctypes.c_int
        result = self.lib.Model_classify(self.model, featureArray)

        return int(result)
    
    def benchmark(self, nTests=1000):
        '''Returns a tuple of the average encode latency and the average
        classify latench in seconds'''

        avgEncodeLatency = ctypes.c_double()
        avgClassifyLatency = ctypes.c_double()

        self.lib.Model_benchmark(
            self.model,
            ctypes.c_int(nTests),
            ctypes.byref(avgEncodeLatency),
            ctypes.byref(avgClassifyLatency)
        )

        return float(avgEncodeLatency.value), float(avgClassifyLatency.value)

    def benchThroughput(self, nTests=1000, nThreads=None):
        if nThreads is None:
            nThreads = os.cpu_count()
        
        encodeThroughput = ctypes.c_double()
        classifyThroughput = ctypes.c_double()

        self.lib.Model_benchThroughput(
            self.model,
            ctypes.c_int(nTests),
            ctypes.c_int(nThreads),
            ctypes.byref(encodeThroughput),
            ctypes.byref(classifyThroughput)
        )

        return float(encodeThroughput.value), float(classifyThroughput.value)
    
    @staticmethod
    def load(modelFn, model=None):        
        Model.lib.Model_load.restype = ctypes.c_void_p

        if model is None:
            model = Model(None, None, None, None, None)

        model.model = Model.lib.Model_load(
            ctypes.c_char_p(modelFn.encode('utf-8')),
        )

        Model.lib.Model_getFeatureSize.restype = ctypes.c_int
        model.featureSize = int(Model.lib.Model_getFeatureSize(model.model))

        return model

    def save(self, modelFn):
        self.lib.Model_save(
            self.model,
            ctypes.c_char_p(modelFn.encode('utf-8'))
        )

    def __del__(self):
        self.lib.Model_delete(self.model)

class MNIST_Model(Model):

    def __init__(self, hypervectorSize, inputQuant, classVectorQuant, imageSize):
        if (
            hypervectorSize is None and
            inputQuant is None and
            classVectorQuant is None and
            imageSize is None
        ):
            return

        assert(imageSize <= 28 and imageSize >= 9)
        Model.__init__(self, hypervectorSize, classVectorQuant,
            inputQuant, imageSize * imageSize, 10)

    
    def train(self, trainSamples=60000, retrainIterations=3):
        assert(trainSamples <= 60000)

        imageSize = int(math.sqrt(self.featureSize))

        labelsFn = f"mnist/train-labels-{imageSize}x{imageSize}-60000.idx1-ubyte"
        imagesFn = f"mnist/train-images-{imageSize}x{imageSize}-60000.idx3-ubyte"
        Model.train(self, trainSamples, retrainIterations, labelsFn, imagesFn)

    def trainOneIteration(self, trainSamples=60000):
        assert(trainSamples <= 60000)

        imageSize = int(math.sqrt(self.featureSize))

        labelsFn = f"mnist/train-labels-{imageSize}x{imageSize}-60000.idx1-ubyte"
        imagesFn = f"mnist/train-images-{imageSize}x{imageSize}-60000.idx3-ubyte"
        Model.trainOneIteration(self, trainSamples, labelsFn, imagesFn)

    def test(self, testSamples=10000):
        assert(testSamples <= 10000)

        imageSize = int(math.sqrt(self.featureSize))

        labelsFn = f"mnist/test-labels-{imageSize}x{imageSize}-10000.idx1-ubyte"
        imagesFn = f"mnist/test-images-{imageSize}x{imageSize}-10000.idx3-ubyte"
        return Model.test(self, testSamples, labelsFn, imagesFn)

    @staticmethod
    def load(modelFn):
        return Model.load(modelFn, MNIST_Model(None, None, None, None))
        
class ISOLET_Model(Model):

    def __init__(self, hypervectorSize, inputQuant, classVectorQuant):
        if (
            hypervectorSize is None and
            inputQuant is None and
            classVectorQuant is None
        ):
            return

        Model.__init__(self, hypervectorSize, classVectorQuant,
            inputQuant, 617, 26)

    def train(self, trainSamples=6238, retrainIterations=3):
        assert(trainSamples <= 6238)

        labelsFn = "isolet/train-labels.idx1-ubyte"
        featuresFn = "isolet/train-features.idx3-ubyte"
        Model.train(self, trainSamples, retrainIterations, labelsFn, featuresFn)
    
    def trainOneIteration(self, trainSamples=6238):
        assert(trainSamples <= 6238)

        labelsFn = "isolet/train-labels.idx1-ubyte"
        featuresFn = "isolet/train-features.idx3-ubyte"
        Model.trainOneIteration(self, trainSamples, labelsFn, featuresFn)

    def test(self, testSamples=1559):
        assert(testSamples <= 1559)

        labelsFn = "isolet/test-labels.idx1-ubyte"
        featuresFn = "isolet/test-features.idx3-ubyte"
        return Model.test(self, testSamples, labelsFn, featuresFn)

    @staticmethod
    def load(modelFn):
        return Model.load(modelFn, ISOLET_Model(None, None, None))
