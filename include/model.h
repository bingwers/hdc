
#ifndef HDC_MODEL_H
#define HDC_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hypervector.h"

#define N_THREADS (8)

typedef struct Model Model;

struct Model {
    Hypervector_Basis basis;
    Hypervector_ClassifySet classifySet;
    size_t downsize;
    size_t featureSize;
    size_t classVecQuant;
    Hypervector_TrainSet tmpTrainSet;
    bool tmpTrainSetValid;
};

void Model_save(Model * model, const char * modelFn);

Model * Model_load(const char * modelFn);

Model * Model_new(int hypervectorSize, int inputQuant, int classVectorQuant,
    int featureSize, int nLabels);

int Model_getFeatureSize(Model * model);

void Model_train(Model * model, const char * labelsFn, const char * featuresFn,
    int trainSamples, int retrainIterations);

void Model_trainOneIteration(Model * model, const char * labelsFn, const char * featuresFn,
    int numTrain);

int Model_classify(Model * model, uint8_t * feature);

int Model_test(Model * model, const char * labelsFn, const char * featuresFn,
    int testSamples);

void Model_delete(Model * model);

#endif // HDC_MODEL_H