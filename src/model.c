
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "model.h"
#include "dataset.h"
#include "hypervector.h"

struct TrainJob {
    Hypervector_Basis * basis;
    Hypervector_TrainSet * trainSet;
    Hypervector_ClassifySet * classifySet;
    uint8_t * labels;
    uint8_t ** features;
    bool retrain;
    pthread_mutex_t * mutex;
    size_t startFeature;
    size_t endFeature;
    int * nWrong;
};

struct TestJob {
    Hypervector_ClassifySet * classifySet;
    Hypervector_Basis * basis;
    uint8_t ** features;
    uint8_t * labels;
    int localNCorrect;
    size_t featureStart;
    size_t featureEnd;
};

void * parallelTrainFunc(void * arg) {
    Hypervector_Basis * basis = ((struct TrainJob*)arg) -> basis;
    Hypervector_TrainSet * trainSet = ((struct TrainJob*)arg) -> trainSet;
    Hypervector_ClassifySet * classifySet = ((struct TrainJob*)arg) -> classifySet;
    uint8_t * labels = ((struct TrainJob*)arg) -> labels;
    uint8_t ** features = ((struct TrainJob*)arg) -> features;
    bool retrain = ((struct TrainJob*)arg) -> retrain;
    pthread_mutex_t * mutex = ((struct TrainJob*)arg) -> mutex;
    size_t startFeature = ((struct TrainJob*)arg) -> startFeature;
    size_t endFeature = ((struct TrainJob*)arg) -> endFeature;
    uint32_t * nWrong = ((struct TrainJob*)arg) -> nWrong;

    size_t i; for (i = startFeature; i < endFeature; i++) {
        Hypervector_Hypervector vector = hypervector_encode(features[i], basis);

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
    uint8_t ** features,
    bool retrain,
    size_t nItems
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
        trainJobs[i].features = features;
        trainJobs[i].retrain = retrain;
        trainJobs[i].mutex = &mutex;
        trainJobs[i].nWrong = &nWrong;

        trainJobs[i].startFeature = nItems * i / N_THREADS;
        if (i == N_THREADS - 1) {
            trainJobs[i].endFeature = nItems;
        }
        else {
            trainJobs[i].endFeature = nItems * (i + 1) / N_THREADS;
        }
    }

    for (i = 0; i < N_THREADS; i++) {
        pthread_create(&threads[i], NULL, parallelTrainFunc, &trainJobs[i]);
    }

    for (i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return (nItems - nWrong);
}

void trainAndRetrain(Hypervector_Basis * basis,
    Hypervector_ClassifySet * classifySet, uint8_t ** features,
    uint8_t * labels, size_t nItems, size_t featureSize,
    size_t numTrain, int numRetrain, int quantization) {

    size_t hypervectorSize = basis -> basisVectors[0].length;

    Hypervector_TrainSet trainSet;
    hypervector_newTrainSet(&trainSet, hypervectorSize, classifySet -> nLabels);
    hypervector_deleteClassifySet(classifySet);

    if (numTrain > nItems) {
        numTrain = nItems;
    }

    size_t i;

    // Training
    //printf("Training... "); fflush(stdout);
    parallelTrain(basis, &trainSet, classifySet, labels, features, false, numTrain);
    //printf("done\n");
    hypervector_newClassifySet(classifySet, &trainSet, quantization);

    // Retraining
    int r; for (r = 0; r < numRetrain; r++) {
        //printf("Retraining %d/%d... ", r+1, numRetrain); fflush(stdout);
        int nCorrect = parallelTrain(basis, &trainSet, classifySet, labels,
                                            features, true, numTrain);
        //printf("done (last iteration %d/%d correct)\n", (int)nCorrect, (int)numTrain);


        hypervector_deleteClassifySet(classifySet);
        hypervector_newClassifySet(classifySet, &trainSet, quantization);
    }

    hypervector_deleteTrainSet(&trainSet);
}

void * parallelTestFunc(void * arg) {
    struct TestJob * testJob = (struct TestJob *)arg;

    Hypervector_ClassifySet * classifySet = testJob -> classifySet;
    Hypervector_Basis * basis = testJob -> basis;
    uint8_t ** features = testJob -> features;
    uint8_t * labels = testJob -> labels;
    size_t featureStart = testJob -> featureStart;
    size_t featureEnd = testJob -> featureEnd;

    size_t i; for (i = featureStart; i < featureEnd; i++) {
        Hypervector_Hypervector vector = hypervector_encode(features[i], basis);

        size_t label = hypervector_classify(classifySet, &vector);
        if ((int)labels[i] == (int)label) {
            testJob -> localNCorrect++;
        }

        hypervector_deleteVector(&vector);
    }

    return NULL;
}

int test(Hypervector_ClassifySet * classifySet, Hypervector_Basis * basis,
    uint8_t ** features, uint8_t * labels, size_t nItems, int nTestSamples) {

    int nTest = nTestSamples;
    if (nTest > nItems) {
        nTest = nItems;
    }

    struct TestJob testJobs[N_THREADS];
    pthread_t threads[N_THREADS];

    //printf("Testing..."); fflush(stdout);

    int i;
    for (i = 0; i < N_THREADS; i++) {
        testJobs[i].classifySet = classifySet;
        testJobs[i].basis = basis;
        testJobs[i].features = features;
        testJobs[i].labels = labels;
        testJobs[i].featureStart = nTest * i / N_THREADS;
        testJobs[i].featureEnd = nTest * (i + 1) / N_THREADS;
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

void Model_save(Model * model, const char * modelFn) {
    FILE * fp = fopen(modelFn, "wb");

    fwrite(&model -> downsize, sizeof(size_t), 1, fp);
    fwrite(&model -> featureSize, sizeof(size_t), 1, fp);
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
    res = fread(&model -> featureSize, sizeof(size_t), 1, fp);
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

Model * Model_new(int hypervectorSize, int inputQuant, int classVectorQuant,
    int featureSize, int nLabels) {
    
    Model * model = (Model*)malloc(sizeof(Model));

    model -> downsize = 1;
    model -> featureSize = featureSize;
    model -> classVecQuant = classVectorQuant / 2;
    model -> tmpTrainSetValid = false;

    hypervector_newBasis(&model -> basis, hypervectorSize, featureSize, inputQuant);
    hypervector_blankClassifySet(&model -> classifySet, nLabels, hypervectorSize);

    return model;
}

int Model_getFeatureSize(Model * model) {
    return (int)model -> featureSize;
}

void Model_train(Model * model, const char * labelsFn, const char * featuresFn,
    int trainSamples, int retrainIterations) {

    Dataset * dataset = Dataset_load(labelsFn, featuresFn, model -> downsize);

    trainAndRetrain(&model -> basis, &model -> classifySet, dataset -> features,
        dataset -> labels, dataset -> nItems, model -> featureSize,
        trainSamples, retrainIterations, model -> classVecQuant);

    Dataset_delete(dataset);
}

void Model_trainOneIteration(Model * model, const char * labelsFn, const char * featuresFn,
    int numTrain) {

    Dataset * dataset = Dataset_load(labelsFn, featuresFn, model -> downsize);
    uint8_t ** features = dataset -> features;
    uint8_t * labels = dataset -> labels;

    size_t length = model -> classifySet.length;
    size_t nLabels = model -> classifySet.nLabels;
    size_t nItems = dataset -> nItems;
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

    if (numTrain > nItems) {
        numTrain = nItems;
    }

    size_t i;

    // Training
    parallelTrain(basis, trainSet, classifySet, labels, features, retrain, numTrain);
    hypervector_deleteClassifySet(classifySet);
    hypervector_newClassifySet(classifySet, trainSet, quantization);

    Dataset_delete(dataset);   
}

int Model_classify(Model * model, uint8_t * feature) {

    Hypervector_Hypervector vector = hypervector_encode(feature, &model -> basis);
    int classification = hypervector_classify(&model -> classifySet, &vector);
    hypervector_deleteVector(&vector);

    return classification;
}

int Model_test(Model * model, const char * labelsFn, const char * featuresFn,
    int testSamples) {

    Dataset * dataset = Dataset_load(labelsFn, featuresFn, model -> downsize);
    
    int nCorrect = test(&model -> classifySet, &model -> basis, dataset -> features,
        dataset -> labels, dataset -> nItems, testSamples);

    Dataset_delete(dataset);

    return nCorrect;
}

void Model_benchmark(Model * model, int nTests, double * avgEncodeLatency,
    double * avgClassifyTime) {

    uint8_t ** features = malloc(sizeof(uint8_t*) * nTests);
    int featuresSize = model -> featureSize;
    int i; for (i = 0; i < nTests; i++) {
        features[i] = malloc(sizeof(uint8_t) * featuresSize);
        
        int j; for (j = 0; j < featuresSize; j++) {
            features[i][j] = rand() & 0xFF;
        }
    }
    Hypervector_Hypervector * vectors =
        malloc(sizeof(Hypervector_Hypervector) * nTests);

    // encode test
    clock_t start, end;
    start = clock();

    for (i = 0; i < nTests; i++) {
        Hypervector_Hypervector vector = hypervector_encode(features[i], &model -> basis);
        vectors[i] = vector;
    }

    end = clock();
    double totalEncodeTime = (double)(end - start) / CLOCKS_PER_SEC;
    *avgEncodeLatency = totalEncodeTime / nTests;

    // classify
    start = clock();

    for (i = 0; i < nTests; i++) {
        int class = hypervector_classify(&model -> classifySet, &vectors[i]);
    }

    end = clock();
    double totalClassifyTime = (double)(end - start) / CLOCKS_PER_SEC;
    *avgClassifyTime = totalClassifyTime / nTests;

    // clean up
    for (i = 0; i < nTests; i++) {
        free(features[i]);
        hypervector_deleteVector(&vectors[i]);
    }
    free(features);
    free(vectors);
}

void Model_delete(Model * model) {
    hypervector_deleteBasis(&model -> basis);
    hypervector_deleteClassifySet(&model -> classifySet);

    if (model -> tmpTrainSetValid) {
        hypervector_deleteTrainSet(&model -> tmpTrainSet);
    }
}

