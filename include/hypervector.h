

#ifndef HDC_HYPERVECTOR_H
#define HDC_HYPERVECTOR_H

#include <stdint.h>
#include <stddef.h>

typedef struct Hypervector_Basis Hypervector_Basis;
typedef struct Hypervector_Hypervector Hypervector_Hypervector;
typedef struct Hypervector_TrainSet Hypervector_TrainSet;
typedef struct Hypervector_ClassifySet Hypervector_ClassifySet;

struct Hypervector_Hypervector {
    size_t length;
    uint8_t * elems; // bit array
};

struct Hypervector_Basis {
    size_t nInputs;
    size_t nLevels;
    Hypervector_Hypervector * basisVectors;
    Hypervector_Hypervector * levelVectors;
};

struct Hypervector_TrainSet {
    size_t nLabels;
    size_t length;
    int32_t ** vectors;
    size_t nTrainSamples;
};

struct Hypervector_ClassifySet {
    size_t nLabels;
    size_t length;
    int32_t ** classVectors;
    double * vectorLengths;
};

void hypervector_newVector(Hypervector_Hypervector * vector, size_t length);

void hypervector_xorVector(Hypervector_Hypervector * dest,
    Hypervector_Hypervector * src1, Hypervector_Hypervector * src2);

void hypervector_deleteVector(Hypervector_Hypervector * vector);

void hypervector_newBasis(Hypervector_Basis * basis, size_t length,
    size_t nInputs, size_t nLevels);

void hypervector_deleteBasis(Hypervector_Basis * basis);

Hypervector_Hypervector hypervector_encode(uint8_t * input, Hypervector_Basis * basis);

void hypervector_newTrainSet(Hypervector_TrainSet * trainSet, size_t length, size_t nLabels);

void hypervector_deleteTrainSet(Hypervector_TrainSet * trainSet);

void hypervector_train(Hypervector_TrainSet * trainSet, Hypervector_Hypervector * vector, 
    size_t label);

void hypervector_untrain(Hypervector_TrainSet * trainSet, Hypervector_Hypervector * vector, 
    size_t label);

void hypervector_newClassifySet(Hypervector_ClassifySet * classifySet,
    Hypervector_TrainSet * trainSet, int quantize);

void hypervector_blankClassifySet(Hypervector_ClassifySet * classifySet,
    size_t nLabels, size_t length);

void hypervector_deleteClassifySet(Hypervector_ClassifySet * classifySet);

size_t hypervector_classify(Hypervector_ClassifySet * classifySet,
    Hypervector_Hypervector * vector);

#endif // HDC_HYPERVECTOR_H