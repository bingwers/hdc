
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist.c"
#include "hypervector.c"

#define HYPERVECTOR_SIZE (10000)
#define N_LABELS (10)

Hypervector_ClassifySet train(Hypervector_Basis * basis, uint8_t ** images,
    uint8_t * labels, size_t nImages, size_t imageSize) {

    Hypervector_TrainSet trainSet;
    hypervector_newTrainSet(&trainSet, HYPERVECTOR_SIZE, N_LABELS);

    size_t numTrain = 10000;
    if (numTrain > nImages) {
        numTrain = nImages;
    }

    size_t i; for (i = 0; i < numTrain; i++) {
        int16_t * encodedVector = hypervector_encode(images[i], basis);

        hypervector_train(&trainSet, encodedVector, labels[i]);

        //hypervector_deleteVector(&vector);
        free(encodedVector);
        
        printf("\rTraining... %d/%d", (int)i, (int)numTrain);
        fflush(stdout);
    }

    Hypervector_ClassifySet classifySet;
    hypervector_newClassifySet(&classifySet, &trainSet);

    hypervector_deleteTrainSet(&trainSet);
    return classifySet;
}

Hypervector_ClassifySet trainAndRetrain(Hypervector_Basis * basis, uint8_t ** images,
    uint8_t * labels, size_t nImages, size_t imageSize) {

    Hypervector_TrainSet trainSet;
    hypervector_newTrainSet(&trainSet, HYPERVECTOR_SIZE, N_LABELS);

    size_t numTrain = 60000;
    if (numTrain > nImages) {
        numTrain = nImages;
    }

    size_t i; for (i = 0; i < numTrain; i++) {
        //Hypervector_Hypervector vector = hypervector_encode(images[i], basis);
        int16_t * encodedVector = hypervector_encode(images[i], basis);

        //hypervector_train(&trainSet, &vector, labels[i]);
        hypervector_train(&trainSet, encodedVector, labels[i]);

        //hypervector_deleteVector(&vector);
        free(encodedVector);
        
        printf("\rTraining... %d/%d", (int)i, (int)numTrain);
        fflush(stdout);
    }

    Hypervector_ClassifySet classifySet;
    hypervector_newClassifySet(&classifySet, &trainSet);

    int r; for (r = 0; r < 1; r++) {
        int nCorrect = 0;

        size_t i; for (i = 0; i < numTrain; i++) {
            //Hypervector_Hypervector vector = hypervector_encode(images[i], basis);
            int16_t * encodedVector = hypervector_encode(images[i], basis);

            size_t classification = hypervector_classify(&classifySet, encodedVector);
            if (classification != (size_t)labels[i]) {
                hypervector_train(&trainSet, encodedVector, labels[i]);
                hypervector_untrain(&trainSet, encodedVector, classification);
            }
            else {
                nCorrect++;
            }

            //hypervector_deleteVector(vector);
            free(encodedVector);
            
            printf("\rRetraining [%d]... %d/%d", r, (int)i, (int)numTrain);
            fflush(stdout);
        }

        printf("\nAccuracy %d/%d\n", nCorrect, (int)numTrain);

        hypervector_deleteClassifySet(&classifySet);
        hypervector_newClassifySet(&classifySet, &trainSet);
    }

    hypervector_deleteTrainSet(&trainSet);
    return classifySet;
}

void test(Hypervector_ClassifySet * classifySet, Hypervector_Basis * basis,
    uint8_t ** images, uint8_t * labels) {

    int nCorrect = 0;    
    size_t i; for (i = 0; i < 10000; i++) {
        //Hypervector_Hypervector vector = hypervector_encode(images[i], basis);
        int16_t * encodedVector = hypervector_encode(images[i], basis);

        size_t label = hypervector_classify(classifySet, encodedVector);
        printf("Test %d:  %d %d\n", (int)i, (int)labels[i], (int)label);

        if ((int)labels[i] == (int)label) {
            nCorrect++;
        }

        //hypervector_deleteVector(&vector);
        free(encodedVector);
    }

    printf("\nNumber Correct: %d\n", nCorrect);
}

int main() {
    unsigned int nTrainItems, nTrainImages, width, height;
    unsigned char * trainLabels = mnist_loadLabels("mnist/train-labels.idx1-ubyte",
        &nTrainItems);
    unsigned char ** trainImages = mnist_loadImages("mnist/train-images.idx3-ubyte",
        &nTrainImages, &width, &height);
    unsigned char ** downscaledTrainImages = mnist_resizeImages(trainImages,
        nTrainImages, &width, &height, 2);

    unsigned int nTestItems, nTestImages;
    unsigned char * testLabels = mnist_loadLabels("mnist/t10k-labels.idx1-ubyte",
        &nTestItems);
    unsigned char ** testImages = mnist_loadImages("mnist/t10k-images.idx3-ubyte",
        &nTestImages, &width, &height);
    unsigned char ** downscaledTestImages = mnist_resizeImages(testImages,
        nTestImages, &width, &height, 2);

    printf("Loaded images\n");
    unsigned int imageSize = width * height;

    Hypervector_Basis basis;
    hypervector_newBasis(&basis, HYPERVECTOR_SIZE, imageSize);
    
    Hypervector_ClassifySet classifySet =
        //train(&basis, downscaledTrainImages, trainLabels, nTrainImages, imageSize);
        trainAndRetrain(&basis, downscaledTrainImages, trainLabels, nTrainImages, imageSize);

    printf("Training Done\n");

    test(&classifySet, &basis, downscaledTestImages, testLabels);

    hypervector_deleteBasis(&basis);
    hypervector_deleteClassifySet(&classifySet);
    mnist_deleteImages(trainImages, nTrainImages);
    mnist_deleteImages(downscaledTrainImages, nTrainImages);
    free(trainLabels);
    mnist_deleteImages(testImages, nTestImages);
    mnist_deleteImages(downscaledTestImages, nTestImages);
    free(testLabels);

    return 0;
}
