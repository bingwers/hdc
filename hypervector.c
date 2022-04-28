

#ifndef HYPERVECTOR_C
#define HYPERVECTOR_C

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define N_LEVELS (16)
#define LEVEL_DOWNSCALE (256 / N_LEVELS)

typedef struct Hypervector_Basis Hypervector_Basis;
typedef struct Hypervector_Hypervector Hypervector_Hypervector;
typedef struct Hypervector_TrainSet Hypervector_TrainSet;
typedef struct Hypervector_ClassifySet Hypervector_ClassifySet;

struct Hypervector_Hypervector {
    size_t length;
    uint8_t * elems; // bit array
};

struct Hypervector_Basis {
    size_t n;
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
    //Hypervector_Hypervector * classVectors;
    size_t length;
    int32_t ** classVectors;
    double * vectorLengths;
};

void hypervector_newVector(Hypervector_Hypervector * vector, size_t length) {
    vector -> length = length;
    vector -> elems = (uint8_t*)malloc(length / 8 + 8);
}

void hypervector_xorVector(Hypervector_Hypervector * dest,
    Hypervector_Hypervector * src1, Hypervector_Hypervector * src2) {
    
    size_t lengthQwords = dest -> length / 64 + 1;

    uint64_t * destElems = (uint64_t *)dest -> elems;
    uint64_t * src1Elems = (uint64_t *)src1 -> elems;
    uint64_t * src2Elems = (uint64_t *)src2 -> elems;

    size_t i; for (i = 0; i < lengthQwords; i++) {
        destElems[i] = src1Elems[i] ^ src2Elems[i];
    }
}

void hypervector_deleteVector(Hypervector_Hypervector * vector) {
    free(vector -> elems);
}

void hypervector_newBasis(Hypervector_Basis * basis, size_t length, size_t n) {
    size_t lengthBytes = length / 8 + 1;

    basis -> n = n;
    basis -> basisVectors = (Hypervector_Hypervector*)
        malloc(sizeof(Hypervector_Hypervector) * n);

    Hypervector_Hypervector * vector = basis -> basisVectors;
    size_t i; for (i = 0; i < n; i++) {
        hypervector_newVector(vector, length);

        size_t j; for (j = 0; j < lengthBytes; j++) {
            vector -> elems[j] = rand() & 0xFF;
        }

        vector++;
    }

    // set the first level vector
    basis -> levelVectors = (Hypervector_Hypervector*)
        malloc(sizeof(Hypervector_Hypervector) * N_LEVELS);

    {
        hypervector_newVector(&basis -> levelVectors[0], length);
        size_t lengthBytes = length / 8 + 1;
        size_t j; for (j = 0; j < lengthBytes; j++) {
            basis -> levelVectors[0].elems[j] = rand() & 0xFF;
        }
    }

    // construct the other level vectors through bit flips
    size_t flipsPerLevel = length / (2 * (N_LEVELS - 1));
    bool * flippedBits = malloc(sizeof(bool) * length);
    memset(flippedBits, 0, sizeof(bool) * length);
    Hypervector_Hypervector * prevVector = basis -> levelVectors;
    vector = prevVector + 1;
    for (i = 1; i < N_LEVELS; i++) {
        hypervector_newVector(vector, length);

        memcpy(vector -> elems, prevVector -> elems, lengthBytes);

        size_t j; for (j = 0; j < flipsPerLevel; j++) {
            size_t index = rand() % length;

            while (flippedBits[index]) {
                index = (index + 1) % length;
            }

            flippedBits[index] = 1;
            
            bool prevBit = (vector -> elems[index >> 3] >> (index & 7)) & 1;
            if (prevBit) {
                vector -> elems[index >> 3] &= ~(1 << (index & 0x7));
            }
            else {
                vector -> elems[index >> 3] |= (1 << (index & 0x7));
            }
        }

        prevVector = vector;
        vector++;
    }
}

void hypervector_deleteBasis(Hypervector_Basis * basis) {
    size_t i; for (i = 0; i < basis -> n; i++) {
        hypervector_deleteVector(&basis -> basisVectors[i]);
    }
    free(basis -> basisVectors);
    for (i = 0; i < N_LEVELS; i++) {
        hypervector_deleteVector(&basis -> levelVectors[i]);
    }
    free(basis -> levelVectors);
}

uint64_t encodeBitConversionTable[16] = {
    0x0, 0x1, 0x10000, 0x10001,
    0x100000000, 0x100000001, 0x100010000, 0x100010001,
    0x1000000000000, 0x1000000000001, 0x1000000010000, 0x1000000010001,
    0x1000100000000, 0x1000100000001, 0x1000100010000, 0x1000100010001
};

Hypervector_Hypervector hypervector_encode(uint8_t * input, Hypervector_Basis * basis) {
    
    size_t n = basis -> n;
    size_t halfN = n / 2;
    size_t length = basis -> basisVectors[0].length;
    size_t lengthBytes = (length / 8) + 1;

    uint16_t * accBuf = (uint16_t*)malloc(sizeof(uint16_t) * (length + 8));
    memset(accBuf, 0, sizeof(uint16_t) * (length + 8));
    uint64_t * accBuf_as_u64 = (uint64_t *)accBuf;

    Hypervector_Hypervector dotResult; hypervector_newVector(&dotResult, length);

    size_t i; for (i = 0; i < n; i++) {
        Hypervector_Hypervector * levelVector = &basis -> levelVectors[input[i] / LEVEL_DOWNSCALE];
        Hypervector_Hypervector * basisVector = &basis -> basisVectors[i];
        
        hypervector_xorVector(&dotResult, levelVector, basisVector);

        uint8_t * bitArray = dotResult.elems;
        //memset(bitArray, 0, lengthBytes);

        // 28s
        /*size_t j; for (j = 0; j < length; j++) {
            uint16_t elem = (bitArray[j >> 3] >> (j & 0x7)) & 1;

            // optimization note: before this was a bipolar add for simplicity
            // so it would do accBuf[j] += elem ? 1 : -1
            // I thought it would compile into a comv, but appearently even on
            // -O4 gcc prefers to make it a branch; unfortunately by design
            // the hypervectors have an equal distribution of 1s and 0s so this
            // prdocued a branch that was the worst case scenario for the branch
            // predictor and it KILLED performance!!!! (112s vs 28s)
            accBuf[j] += elem;
        }*/

        // 19.7s
        /*size_t j; for (j = 0; j < lengthBytes; j++) {
            uint8_t val = bitArray[j];
            accBuf[8*j + 0] += val & 1;
            accBuf[8*j + 1] += (val >> 1) & 1;
            accBuf[8*j + 2] += (val >> 2) & 1;
            accBuf[8*j + 3] += (val >> 3) & 1;
            accBuf[8*j + 4] += (val >> 4) & 1;
            accBuf[8*j + 5] += (val >> 5) & 1;
            accBuf[8*j + 6] += (val >> 6) & 1;
            accBuf[8*j + 7] += (val >> 7) & 1;
        }*/

        // 8.5s
        /*size_t j; for (j = 0; j < lengthBytes; j++) {
            uint8_t val = bitArray[j];

            uint64_t low = (val & 1)
                            + ((uint64_t)(val & 2) <<  15)
                            + ((uint64_t)(val & 4) <<  30)
                            + ((uint64_t)(val & 8) <<  45);
            uint64_t hi  = ((uint64_t)(val & 16) >> 4)
                            + ((uint64_t)(val & 32) <<  11)
                            + ((uint64_t)(val & 64) <<  26)
                            + ((uint64_t)(val & 128) <<  41);

            accBuf_as_u64[2 * j] += low;
            accBuf_as_u64[2 * j + 1] += hi;
        }*/

        // 7.8s
        size_t j; for (j = 0; j < lengthBytes; j++) {
            uint8_t val = bitArray[j];

            uint64_t low = encodeBitConversionTable[val & 0xF];
            uint64_t hi = encodeBitConversionTable[val >> 4];

            accBuf_as_u64[2 * j] += low;
            accBuf_as_u64[2 * j + 1] += hi;
        }
    }

    //hypervector_deleteVector(&dotResult);

    Hypervector_Hypervector vector = dotResult; // move operation
    //hypervector_newVector(&vector, length);

    size_t j; for (j = 0; j < length; j++) {
        if (accBuf[j] > halfN) {
            vector.elems[j >> 3] |= (1 << (j & 0x7));
        }
        else {
            vector.elems[j >> 3] &= ~(1 << (j & 0x7));
        }
    }

    free(accBuf);

    return vector;
}

void hypervector_newTrainSet(Hypervector_TrainSet * trainSet, size_t length, size_t nLabels) {
    trainSet -> nLabels = nLabels;
    trainSet -> length = length;
    trainSet -> vectors = (int32_t**)malloc(sizeof(int32_t*) * nLabels);
    trainSet -> nTrainSamples = 0;

    size_t i; for (i = 0; i < nLabels; i++) {
        trainSet -> vectors[i] = (int32_t*)malloc(sizeof(int32_t) * length);
        memset(trainSet -> vectors[i], 0, sizeof(int32_t) * length);
    }
}

void hypervector_deleteTrainSet(Hypervector_TrainSet * trainSet) {
    size_t i; for (i = 0; i < trainSet -> nLabels; i++) {
        free(trainSet -> vectors[i]);
    }
    free(trainSet -> vectors);
}

void hypervector_train(Hypervector_TrainSet * trainSet, Hypervector_Hypervector * vector, 
    size_t label) {
    
    size_t length = vector -> length;

    uint8_t * bitArray = vector -> elems;
    int32_t * trainVector = trainSet -> vectors[label];

    size_t i; for (i = 0; i < length; i++) {
        bool elem = (bitArray[i >> 3] >> (i & 0x7)) & 1;
        trainVector[i] += elem ? 1 : -1;
    }

    trainSet -> nTrainSamples++;
}

void hypervector_untrain(Hypervector_TrainSet * trainSet, Hypervector_Hypervector * vector, 
    size_t label) {
    
    size_t length = vector -> length;

    uint8_t * bitArray = vector -> elems;
    int32_t * trainVector = trainSet -> vectors[label];

    size_t i; for (i = 0; i < length; i++) {
        bool elem = (bitArray[i >> 3] >> (i & 0x7)) & 1;
        trainVector[i] += elem ? -1 : +1;
    }

    trainSet -> nTrainSamples++;
}

void hypervector_newClassifySet(Hypervector_ClassifySet * classifySet,
                                Hypervector_TrainSet * trainSet) {
    
    size_t nLabels = trainSet -> nLabels;
    size_t length = trainSet -> length;

    classifySet -> nLabels = nLabels;
    classifySet -> length = length;
    classifySet -> classVectors = (int32_t**)malloc(sizeof(int32_t*) * nLabels);
    classifySet -> vectorLengths = (double*)malloc(sizeof(double) * nLabels);

    size_t i; for (i = 0; i < nLabels; i++) {
        double vectorLength = 0.0;

        int32_t * classVector = (int32_t*)malloc(sizeof(int32_t) * length);
        
        size_t j; for (j = 0; j < length; j++) {
            int32_t val = trainSet -> vectors[i][j];
            classVector[j] = val;
            double dblVal = (double)val;
            vectorLength += dblVal * dblVal;
        }

        classifySet -> classVectors[i] = classVector;
        classifySet -> vectorLengths[i] = sqrtl(vectorLength);
    }
}

void hypervector_deleteClassifySet(Hypervector_ClassifySet * classifySet) {
    size_t i; for (i = 0; i < classifySet -> nLabels; i++) {
        free(classifySet -> classVectors[i]);
    }
    free(classifySet -> classVectors);
    free(classifySet -> vectorLengths);
}

size_t hypervector_classify(Hypervector_ClassifySet * classifySet,
    Hypervector_Hypervector * vector) {
    
    /*size_t bestLabel = (size_t)(-1);
    size_t smallestDistance = (size_t)(-1);

    size_t length = vector -> length;

    size_t label; for (label = 0; label < classifySet -> nLabels; label++) {
        size_t dist = 0;

        uint8_t * bitArray1 = vector -> elems;
        uint8_t * bitArray2 = classifySet -> classVectors[label].elems;

        size_t j; for (j = 0; j < length; j++) {
            uint8_t val1 = (bitArray1[j >> 3] >> (j & 0x7)) & 1;
            uint8_t val2 = (bitArray2[j >> 3] >> (j & 0x7)) & 1;

            if (val1 ^ val2) {
                dist++;
            }
        }

        if (dist < smallestDistance) {
            bestLabel = label;
            smallestDistance = dist;
        }
    }

    return bestLabel;*/

    size_t bestLabel = (size_t)(-1);
    double maxSimilarity = __DBL_MIN__;

    size_t length = vector -> length;

    size_t label; for (label = 0; label < classifySet -> nLabels; label++) {
        int64_t similarity = 0;

        uint8_t * bitArray = vector -> elems;
        int32_t * classVector = classifySet -> classVectors[label];

        size_t j; for (j = 0; j < length; j++) {
            bool polarity = (bitArray[j >> 3] >> (j & 0x7)) & 1;

            if (polarity) {
                similarity += classVector[j];
            }
            else {
                similarity -= classVector[j];
            }
        }

        double scaledSimilarity = (double)similarity / classifySet -> vectorLengths[label];

        if (scaledSimilarity > maxSimilarity) {
            bestLabel = label;
            maxSimilarity = scaledSimilarity;
        }

        //printf("    similarity for %d: %d\n", (int)label, (int)similarity);
    }

    return bestLabel;
}


#endif // HYPERVECTOR_C