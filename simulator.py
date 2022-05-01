from model import Model

fp = open("simResults/summary.csv", 'w')

for imageSize in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
    for hypervectorSize in [2000, 4000, 6000, 8000, 10000]:
        for classVectorQuant in [2, 4, 16, 64, 0]:
            trainImagesFn = f"mnist/train-images-{imageSize}x{imageSize}-60000.idx3-ubyte"
            trainLabelsFn = f"mnist/train-labels-{imageSize}x{imageSize}-60000.idx1-ubyte"
            testImagesFn = f"mnist/test-images-{imageSize}x{imageSize}-60000.idx3-ubyte"
            testLabelsFn = f"mnist/test-labels-{imageSize}x{imageSize}-60000.idx1-ubyte"

            print(f"imageSize: {imageSize}, hypervectorSize: {hypervectorSize}, classVectorQuant: {classVectorQuant}")

            model = Model(hypervectorSize, 2, classVectorQuant, imageSize)

            for i in range(10):
                modelSaveFn = f"simResults/imsz-{imageSize}-hvsz-{hypervectorSize}-cvq-{classVectorQuant}-iters-{i+1}.hvmodel"

                model.trainOneIteration(trainLabelsFn, trainImagesFn, 60000)
                accuracy = model.test(testLabelsFn, testImagesFn, 10000)
                print(f"    {i}: {accuracy}")

                model.save(modelSaveFn)
            
                fp.write(f"{imageSize}, {hypervectorSize}, {classVectorQuant}, {i+1}, {accuracy}\n")
                fp.flush()

fp.close()

