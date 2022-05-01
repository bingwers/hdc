
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist.c"
#include "hypervector.c"
#include "pthread.h"

#define N_LABELS (10)
#define N_THREADS (8)

struct TrainJob {
    Hypervector_Basis * basis;
    Hypervector_TrainSet * trainSet;
    Hypervector_ClassifySet * classifySet;
    uint8_t * labels;
    uint8_t ** images;
    bool retrain;
    pthread_mutex_t * mutex;
    size_t startImage;
    size_t endImage;
    int * nWrong;
};

struct TestJob {
    Hypervector_ClassifySet * classifySet;
    Hypervector_Basis * basis;
    uint8_t ** images;
    uint8_t * labels;
    int localNCorrect;
    size_t imageStart;
    size_t imageEnd;
};

void * parallelTrainFunc(void * arg) {
    Hypervector_Basis * basis = ((struct TrainJob*)arg) -> basis;
    Hypervector_TrainSet * trainSet = ((struct TrainJob*)arg) -> trainSet;
    Hypervector_ClassifySet * classifySet = ((struct TrainJob*)arg) -> classifySet;
    uint8_t * labels = ((struct TrainJob*)arg) -> labels;
    uint8_t ** images = ((struct TrainJob*)arg) -> images;
    bool retrain = ((struct TrainJob*)arg) -> retrain;
    pthread_mutex_t * mutex = ((struct TrainJob*)arg) -> mutex;
    size_t startImage = ((struct TrainJob*)arg) -> startImage;
    size_t endImage = ((struct TrainJob*)arg) -> endImage;
    uint32_t * nWrong = ((struct TrainJob*)arg) -> nWrong;

    size_t i; for (i = startImage; i < endImage; i++) {
        Hypervector_Hypervector vector = hypervector_encode(images[i], basis);

        if (retrain) {
            size_t classification = hypervector_classify(classifySet, &vector);
            if (classification != labels[i]) {
                pthread_mutex_lock(mutex);
                hypervector_train(trainSet, &vector, labels[i]);
                hypervector_untrain(trainSet, &vector, classification);
                pthread_mutex_unlock(mutex);
                (*nWrong)++;
            }
        }
        else {
            pthread_mutex_lock(mutex);
            hypervector_train(trainSet, &vector, labels[i]);
            pthread_mutex_unlock(mutex);
        }

        hypervector_deleteVector(&vector);
    }

    return NULL;
}

int parallelTrain(
    Hypervector_Basis * basis,
    Hypervector_TrainSet * trainSet,
    Hypervector_ClassifySet * classifySet,
    uint8_t * labels,
    uint8_t ** images,
    bool retrain,
    size_t nImages
) {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    struct TrainJob trainJobs[N_THREADS];
    pthread_t threads[N_THREADS];
    int nWrong = 0;

    int i; for (i = 0; i < N_THREADS; i++) {
        trainJobs[i].basis = basis;
        trainJobs[i].trainSet = trainSet;
        trainJobs[i].classifySet = classifySet;
        trainJobs[i].labels = labels;
        trainJobs[i].images = images;
        trainJobs[i].retrain = retrain;
        trainJobs[i].mutex = &mutex;
        trainJobs[i].nWrong = &nWrong;

        trainJobs[i].startImage = nImages * i / N_THREADS;
        if (i == N_THREADS - 1) {
            trainJobs[i].endImage = nImages;
        }
        else {
            trainJobs[i].endImage = nImages * (i + 1) / N_THREADS;
        }
    }

    for (i = 0; i < N_THREADS; i++) {
        pthread_create(&threads[i], NULL, parallelTrainFunc, &trainJobs[i]);
    }

    for (i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return (nImages - nWrong);
}

void trainAndRetrain(Hypervector_Basis * basis,
    Hypervector_ClassifySet * classifySet, uint8_t ** images,
    uint8_t * labels, size_t nImages, size_t imageSize,
    size_t numTrain, int numRetrain, int quantization) {

    size_t hypervectorSize = basis -> basisVectors[0].length;

    Hypervector_TrainSet trainSet;
    hypervector_newTrainSet(&trainSet, hypervectorSize, N_LABELS);
    hypervector_deleteClassifySet(classifySet);

    if (numTrain > nImages) {
        numTrain = nImages;
    }

    size_t i;

    // Training
    printf("Training... "); fflush(stdout);
    parallelTrain(basis, &trainSet, classifySet, labels, images, false, numTrain);
    printf("done\n");
    hypervector_newClassifySet(classifySet, &trainSet, quantization);

    // Retraining
    int r; for (r = 0; r < numRetrain; r++) {
        printf("Retraining %d/%d... ", r+1, numRetrain); fflush(stdout);
        int nCorrect = parallelTrain(basis, &trainSet, classifySet, labels,
                                            images, true, numTrain);
        printf("done (last iteration %d/%d correct)\n", (int)nCorrect, (int)numTrain);


        hypervector_deleteClassifySet(classifySet);
        hypervector_newClassifySet(classifySet, &trainSet, quantization);
    }

    hypervector_deleteTrainSet(&trainSet);
}

void * parallelTestFunc(void * arg) {
    struct TestJob * testJob = (struct TestJob *)arg;

    Hypervector_ClassifySet * classifySet = testJob -> classifySet;
    Hypervector_Basis * basis = testJob -> basis;
    uint8_t ** images = testJob -> images;
    uint8_t * labels = testJob -> labels;
    size_t imageStart = testJob -> imageStart;
    size_t imageEnd = testJob -> imageEnd;

    size_t i; for (i = imageStart; i < imageEnd; i++) {
        Hypervector_Hypervector vector = hypervector_encode(images[i], basis);

        size_t label = hypervector_classify(classifySet, &vector);
        if ((int)labels[i] == (int)label) {
            testJob -> localNCorrect++;
        }

        hypervector_deleteVector(&vector);
    }

    return NULL;
}

int test(Hypervector_ClassifySet * classifySet, Hypervector_Basis * basis,
    uint8_t ** images, uint8_t * labels, size_t nImages, int nTestSamples) {

    int nTest = nTestSamples;
    if (nTest > nImages) {
        nTest = nImages;
    }

    struct TestJob testJobs[N_THREADS];
    pthread_t threads[N_THREADS];

    //printf("Testing..."); fflush(stdout);

    int i;
    for (i = 0; i < N_THREADS; i++) {
        testJobs[i].classifySet = classifySet;
        testJobs[i].basis = basis;
        testJobs[i].images = images;
        testJobs[i].labels = labels;
        testJobs[i].imageStart = nTest * i / N_THREADS;
        testJobs[i].imageEnd = nTest * (i + 1) / N_THREADS;
        testJobs[i].localNCorrect = 0;
    }

    for (i = 0; i < N_THREADS; i++) {
        pthread_create(&threads[i], NULL, parallelTestFunc, &testJobs[i]);
    }

    int nCorrect = 0;
    for (i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
        nCorrect += testJobs[i].localNCorrect;
    }

    //printf("done\nNumber Correct: %d\n", nCorrect);
    return nCorrect;
}

typedef struct ImageData ImageData;
typedef struct Model Model;

struct ImageData {
    unsigned int nImages;
    unsigned int width, height;
    uint8_t * labels;
    uint8_t ** images;
};

struct Model {
    Hypervector_Basis basis;
    Hypervector_ClassifySet classifySet;
    size_t downsize;
    size_t imageSize;
    size_t classVecQuant;
    Hypervector_TrainSet tmpTrainSet;
    bool tmpTrainSetValid;
};

void Model_save(Model * model, const char * modelFn) {
    FILE * fp = fopen(modelFn, "wb");

    fwrite(&model -> downsize, sizeof(size_t), 1, fp);
    fwrite(&model -> imageSize, sizeof(size_t), 1, fp);
    fwrite(&model -> classVecQuant, sizeof(size_t), 1, fp);

    fwrite(&model -> basis.nInputs, sizeof(size_t), 1, fp);
    fwrite(&model -> basis.nLevels, sizeof(size_t), 1, fp);
    fwrite(&model -> classifySet.nLabels, sizeof(size_t), 1, fp);
    fwrite(&model -> classifySet.length, sizeof(size_t), 1, fp);

    size_t length = model -> classifySet.length;
    size_t lengthBytes = length / 8 + 1;
    size_t nLabels = model -> classifySet.nLabels;

    int i;
    for (i = 0; i < model -> basis.nInputs; i++) {
        fwrite(model -> basis.basisVectors[i].elems, 1, lengthBytes, fp);
    }
    for (i = 0; i < model -> basis.nLevels; i++) {
        fwrite(model -> basis.levelVectors[i].elems, 1, lengthBytes, fp);
    }
    for (i = 0; i < model -> classifySet.nLabels; i++) {
        fwrite(model -> classifySet.classVectors[i], sizeof(int32_t), length, fp);
        fwrite(&model -> classifySet.vectorLengths[i], sizeof(double), 1, fp);
    }

    fclose(fp);
}

Model * Model_load(const char * modelFn) {
    FILE * fp = fopen(modelFn, "rb");

    Model * model = (Model*)malloc(sizeof(Model));
    size_t res;

    res = fread(&model -> downsize, sizeof(size_t), 1, fp);
    res = fread(&model -> imageSize, sizeof(size_t), 1, fp);
    res = fread(&model -> classVecQuant, sizeof(size_t), 1, fp);

    res = fread(&model -> basis.nInputs, sizeof(size_t), 1, fp);
    res = fread(&model -> basis.nLevels, sizeof(size_t), 1, fp);
    res = fread(&model -> classifySet.nLabels, sizeof(size_t), 1, fp);
    res = fread(&model -> classifySet.length, sizeof(size_t), 1, fp);

    size_t length = model -> classifySet.length;
    size_t lengthBytes = length / 8 + 1;
    size_t nLabels = model -> classifySet.nLabels;

    model -> basis.basisVectors = (Hypervector_Hypervector*)malloc(
        sizeof(Hypervector_Hypervector) * model -> basis.nInputs);
    model -> basis.levelVectors = (Hypervector_Hypervector*)malloc(
        sizeof(Hypervector_Hypervector) * model -> basis.nLevels);
    model -> classifySet.classVectors = (int32_t **)malloc(
        sizeof(int32_t *) * model -> classifySet.nLabels);
    model -> classifySet.vectorLengths = (double *)malloc(
        sizeof(double) * model -> classifySet.nLabels);

    int i;
    for (i = 0; i < model -> basis.nInputs; i++) {
        hypervector_newVector(&model -> basis.basisVectors[i], length);
        res = fread(model -> basis.basisVectors[i].elems, 1, lengthBytes, fp);
    }
    for (i = 0; i < model -> basis.nLevels; i++) {
        hypervector_newVector(&model -> basis.levelVectors[i], length);
        res = fread(model -> basis.levelVectors[i].elems, 1, lengthBytes, fp);
    }
    for (i = 0; i < model -> classifySet.nLabels; i++) {
        model -> classifySet.classVectors[i] = (int32_t*)malloc(
            sizeof(int32_t) * length);
        res = fread(model -> classifySet.classVectors[i], sizeof(int32_t), length, fp);
        res = fread(&model -> classifySet.vectorLengths[i], sizeof(double), 1, fp);
    }

    fclose(fp);

    model -> tmpTrainSetValid = false;

    return model;
}

ImageData * ImageData_load(const char * labelsFn,
    const char * imagesFn, size_t downscale) {

    ImageData * imageData = (ImageData*)malloc(sizeof(ImageData));

    imageData -> labels = mnist_loadLabels(labelsFn, &imageData -> nImages);
    uint8_t ** rawImages = mnist_loadImages(imagesFn, &imageData -> nImages,
        &imageData -> width, &imageData -> height);
    
    imageData -> images = mnist_resizeImages(rawImages, imageData -> nImages,
        &imageData -> width, &imageData -> height, downscale);
    
    mnist_deleteImages(rawImages, imageData -> nImages);
    return imageData;
}

void ImageData_delete(ImageData * imageData) {
    mnist_deleteImages(imageData -> images, imageData -> nImages);
    free(imageData -> labels);
    free(imageData);
}

Model * Model_new(int hypervectorSize, int inputQuant, int classVectorQuant,
    int imageSizeOneDimension) {
    
    Model * model = (Model*)malloc(sizeof(Model));

    size_t imageSize = imageSizeOneDimension * imageSizeOneDimension;
    model -> downsize = 1;
    model -> imageSize = imageSize;
    model -> classVecQuant = classVectorQuant / 2;
    model -> tmpTrainSetValid = false;

    hypervector_newBasis(&model -> basis, hypervectorSize, imageSize, inputQuant);
    hypervector_blankClassifySet(&model -> classifySet, N_LABELS, hypervectorSize);

    return model;
}

int Model_getImageSize(Model * model) {
    int modelSizeOneDimension = (int)round(sqrt((double)model -> imageSize));

    return modelSizeOneDimension;
}

void Model_train(Model * model, const char * labelsFn, const char * imagesFn,
    int trainSamples, int retrainIterations) {

    ImageData * imageData = ImageData_load(labelsFn, imagesFn, model -> downsize);

    trainAndRetrain(&model -> basis, &model -> classifySet, imageData -> images,
        imageData -> labels, imageData -> nImages, model -> imageSize,
        trainSamples, retrainIterations, model -> classVecQuant);

    ImageData_delete(imageData);
}

void Model_trainOneIteration(Model * model, const char * labelsFn, const char * imagesFn,
    int numTrain) {

    ImageData * imageData = ImageData_load(labelsFn, imagesFn, model -> downsize);
    uint8_t ** images = imageData -> images;
    uint8_t * labels = imageData -> labels;

    size_t length = model -> classifySet.length;
    size_t nLabels = model -> classifySet.nLabels;
    size_t nImages = imageData -> nImages;
    size_t quantization = model -> classVecQuant;

    Hypervector_Basis * basis = &model -> basis;
    Hypervector_TrainSet * trainSet = &model -> tmpTrainSet;
    Hypervector_ClassifySet * classifySet = &model -> classifySet;

    bool retrain = true;
    if (!(model -> tmpTrainSetValid)) {
        hypervector_newTrainSet(trainSet, length, nLabels);
        model -> tmpTrainSetValid = true;
        retrain = false;
    }

    if (numTrain > nImages) {
        numTrain = nImages;
    }

    size_t i;

    // Training
    parallelTrain(basis, trainSet, classifySet, labels, images, retrain, numTrain);
    hypervector_deleteClassifySet(classifySet);
    hypervector_newClassifySet(classifySet, trainSet, quantization);

    ImageData_delete(imageData);   
}

int Model_classify(Model * model, uint8_t * image) {

    uint8_t ** images = &image;
    unsigned int width = 28;
    unsigned int height = 28;
    uint8_t ** scaledImages = mnist_resizeImages(images, 1, &width, &height, model -> downsize);

    Hypervector_Hypervector vector = hypervector_encode(*scaledImages, &model -> basis);
    int classification = hypervector_classify(&model -> classifySet, &vector);

    free(*scaledImages);
    free(scaledImages);
    hypervector_deleteVector(&vector);

    return classification;
}

int Model_test(Model * model, const char * labelsFn, const char * imagesFn,
    int testSamples) {

    ImageData * imageData = ImageData_load(labelsFn, imagesFn, model -> downsize);
    
    int nCorrect = test(&model -> classifySet, &model -> basis, imageData -> images,
        imageData -> labels, imageData -> nImages, testSamples);

    ImageData_delete(imageData);

    return nCorrect;
}

void Model_delete(Model * model) {
    hypervector_deleteBasis(&model -> basis);
    hypervector_deleteClassifySet(&model -> classifySet);

    if (model -> tmpTrainSetValid) {
        hypervector_deleteTrainSet(&model -> tmpTrainSet);
    }
}

