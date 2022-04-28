
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist.c"
#include "hypervector.c"
#include "pthread.h"


#define N_LABELS (10)
#define N_THREADS (8)

#define N_TRAIN_SAMPLES (60000)
#define N_TEST_SAMPLES (10000)

#define HYPERVECTOR_SIZE (10000)
#define INPUT_DOWNSCALE_FACTOR (2) // 1,2, or 3 practical
#define INPUT_QUANTIZATION_LEVELS (2) // max = 256
#define CLASS_VECTOR_QUANTIZATION_LEVELS (2) // should be multiple of 2
#define RETRAIN_ITERATIONS (3)


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

Hypervector_ClassifySet trainAndRetrain(Hypervector_Basis * basis, uint8_t ** images,
    uint8_t * labels, size_t nImages, size_t imageSize) {

    Hypervector_TrainSet trainSet;
    Hypervector_ClassifySet classifySet;
    hypervector_newTrainSet(&trainSet, HYPERVECTOR_SIZE, N_LABELS);

    size_t numTrain = N_TRAIN_SAMPLES;
    int numRetrain = RETRAIN_ITERATIONS;
    int quantization = CLASS_VECTOR_QUANTIZATION_LEVELS / 2;

    if (numTrain > nImages) {
        numTrain = nImages;
    }

    size_t i;

    // Training
    printf("Training... "); fflush(stdout);
    parallelTrain(basis, &trainSet, &classifySet, labels, images, false, numTrain);
    printf("done\n");
    hypervector_newClassifySet(&classifySet, &trainSet, quantization);

    // Retraining
    int r; for (r = 0; r < numRetrain; r++) {
        printf("Retraining %d/%d... ", r+1, numRetrain); fflush(stdout);
        int nCorrect = parallelTrain(basis, &trainSet, &classifySet, labels,
                                            images, true, numTrain);
        printf("done (last iteration %d/%d correct)\n", (int)nCorrect, (int)numTrain);


        hypervector_deleteClassifySet(&classifySet);
        hypervector_newClassifySet(&classifySet, &trainSet, quantization);
    }

    hypervector_deleteTrainSet(&trainSet);
    return classifySet;
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

void test(Hypervector_ClassifySet * classifySet, Hypervector_Basis * basis,
    uint8_t ** images, uint8_t * labels, size_t nImages) {

    int nTest = N_TEST_SAMPLES;
    if (nTest > nImages) {
        nTest = nImages;
    }

    struct TestJob testJobs[N_THREADS];
    pthread_t threads[N_THREADS];

    printf("Testing..."); fflush(stdout);

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

    printf("done\nNumber Correct: %d\n", nCorrect);
}

int main() {
    unsigned int nTrainItems, nTrainImages, width, height;
    unsigned char * trainLabels = mnist_loadLabels("mnist/train-labels.idx1-ubyte",
        &nTrainItems);
    unsigned char ** trainImages = mnist_loadImages("mnist/train-images.idx3-ubyte",
        &nTrainImages, &width, &height);
    unsigned char ** downscaledTrainImages = mnist_resizeImages(trainImages,
        nTrainImages, &width, &height, INPUT_DOWNSCALE_FACTOR);

    unsigned int nTestItems, nTestImages;
    unsigned char * testLabels = mnist_loadLabels("mnist/t10k-labels.idx1-ubyte",
        &nTestItems);
    unsigned char ** testImages = mnist_loadImages("mnist/t10k-images.idx3-ubyte",
        &nTestImages, &width, &height);
    unsigned char ** downscaledTestImages = mnist_resizeImages(testImages,
        nTestImages, &width, &height, INPUT_DOWNSCALE_FACTOR);

    printf("Loaded images\n");
    unsigned int imageSize = width * height;

    Hypervector_Basis basis;
    hypervector_newBasis(&basis, HYPERVECTOR_SIZE, imageSize, INPUT_QUANTIZATION_LEVELS);
    
    Hypervector_ClassifySet classifySet =
        trainAndRetrain(&basis, downscaledTrainImages, trainLabels, nTrainImages, imageSize);

    printf("Training Done\n");

    test(&classifySet, &basis, downscaledTestImages, testLabels, nTestImages);

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
