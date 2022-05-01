
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
            imageSize is None
        ):
            return

        assert(imageSize >= 9 and imageSize <= 28)

        self.imageSize = imageSize
        self.lib.Model_new.restype = ctypes.c_void_p
        self.model = self.lib.Model_new(
            ctypes.c_int(hypervectorSize),
            ctypes.c_int(inputQuant),
            ctypes.c_int(classVectorQuant),
            ctypes.c_int(imageSize)
        )
    
    def train(self, trainSamples=60000, retrainIterations=3, labelsFn=None, imagesFn=None):
        if labelsFn is None:
            labelsFn = f"mnist/train-labels-{self.imageSize}x{self.imageSize}-60000.idx1-ubyte"
        if imagesFn is None:
            imagesFn = f"mnist/train-images-{self.imageSize}x{self.imageSize}-60000.idx3-ubyte"

        self.lib.Model_train(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(imagesFn.encode('utf-8')),
            ctypes.c_int(trainSamples),
            ctypes.c_int(retrainIterations)
        )

    def trainOneIteration(self, trainSamples=60000, labelsFn=None, imagesFn=None):
        if labelsFn is None:
            labelsFn = f"mnist/train-labels-{self.imageSize}x{self.imageSize}-60000.idx1-ubyte"
        if imagesFn is None:
            imagesFn = f"mnist/train-images-{self.imageSize}x{self.imageSize}-60000.idx3-ubyte"

        self.lib.Model_trainOneIteration(
            self.model,
            ctypes.c_char_p(labelsFn.encode('utf-8')),
            ctypes.c_char_p(imagesFn.encode('utf-8')),
            ctypes.c_int(trainSamples)
        )


    def test(self, testSamples=10000, labelsFn=None, imagesFn=None):
        if labelsFn is None:
            labelsFn = f"mnist/test-labels-{self.imageSize}x{self.imageSize}-10000.idx1-ubyte"
        if imagesFn is None:
            imagesFn = f"mnist/test-images-{self.imageSize}x{self.imageSize}-10000.idx3-ubyte"

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

        Model.lib.Model_getImageSize(model.model)
        model.imageSize = int(Model.lib.Model_getImageSize(model.model))

        return model

    def save(self, modelFn):
        self.lib.Model_save(
            self.model,
            ctypes.c_char_p(modelFn.encode('utf-8'))
        )

    def __del__(self):
        self.lib.Model_delete(self.model)
