
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "dataset.h"

uint8_t imageManip_sample(uint8_t * image, int size, float x, float y) {
    if (fabsf(x) > 1.0 || fabs(y) > 1.0) {
        return 0;
    }

    float scaleX = ((x + 1) * 0.5) * (float)size - 0.5;
    float scaleY = ((y + 1) * 0.5) * (float)size - 0.5;

    float refX = floorf(scaleX);
    float refY = floorf(scaleY);
    int brXInt = (int)refX;
    int brYInt = (int)refY;

    float totalWeight = 0;
    float value = 0;
    float weight;

    weight = (1 - fabs(scaleX - refX)) * (1 - fabs(scaleY - refY));
    totalWeight += weight;
    if (brXInt >= 0 && brXInt < size && brYInt >= 0 && brYInt < size) {
        value += weight * (float)image[brYInt * size + brXInt];
    }

    refX += 1; brXInt++;

    weight = (1 - fabs(scaleX - refX)) * (1 - fabs(scaleY - refY));
    totalWeight += weight;
    if (brXInt >= 0 && brXInt < size && brYInt >= 0 && brYInt < size) {
        value += weight * (float)image[brYInt * size + brXInt];
    }

    refY += 1; brYInt++;

    weight = (1 - fabs(scaleX - refX)) * (1 - fabs(scaleY - refY));
    totalWeight += weight;
    if (brXInt >= 0 && brXInt < size && brYInt >= 0 && brYInt < size) {
        value += weight * (float)image[brYInt * size + brXInt];
    }

    refX -= 1; brXInt--;

    weight = (1 - fabs(scaleX - refX)) * (1 - fabs(scaleY - refY));
    totalWeight += weight;
    if (brXInt >= 0 && brXInt < size && brYInt >= 0 && brYInt < size) {
        value += weight * (float)image[brYInt * size + brXInt];
    }

    value /= totalWeight;
    if (value > 255) {
        value ==255;
    }

    return (uint8_t)value;
}

void imageManip_disp(uint8_t * image, size_t size) {
    
    int i;
    int totalSize = size * size;
    printf("+"); for (i = 0; i < 2*size; i++) {printf("-");} printf("+\n");
    for (i = 0; i < totalSize; i++) {
        if (i % size == 0) {
            printf("|");
        }

        if (image[i] < 64) {
            printf("  ");
        }
        else if (image[i] < 128) {
            printf("..");
        }
        else if (image[i] < 192) {
            printf("xx");
        }
        else {
            printf("@@");
        }

        if ((i+1) % size == 0) {
            printf("|\n");
        }
    }
    printf("+"); for (i = 0; i < 2*size; i++) {printf("-");} printf("+\n");
}

uint8_t * imageManip_upsize(uint8_t * image, size_t size, size_t newSize) {
    uint8_t * newImage = (uint8_t *)malloc(newSize * newSize);

    float scale = (float)2 / (float)newSize;

    int i;
    int j;
    for (j = 0; j < newSize; j++) {
        float y = ((float)j + 0.5) * scale - 1;
        for (i = 0; i < newSize; i++) {
            float x = ((float)i + 0.5) * scale - 1;

            newImage[newSize*j + i] = imageManip_sample(image, size, x, y);
        }
    }

    return newImage;
}

uint8_t * imageManip_downsize(uint8_t * image, size_t size, size_t newSize) {
    const int upscale = 5;
    uint8_t * upscaled = imageManip_upsize(image, size, size * upscale);

    size_t upsize = upscale * newSize;
    float scale = (float)2 / (float)upsize;

    uint32_t * acc = (uint32_t*)malloc(sizeof(uint32_t) * newSize * newSize);
    uint8_t * newImage = (uint8_t*)malloc(newSize * newSize);
    memset(acc, 0, sizeof(uint32_t) * newSize * newSize);

    int i;
    int j;
    for (j = 0; j < upsize; j++) {
        float y = ((float)j + 0.5) * scale - 1;
        int yDown = j / upscale;
        for (i = 0; i < upsize; i++) {
            float x = ((float)i + 0.5) * scale - 1;
            int xDown = i / upscale;

            acc[yDown*newSize + xDown] += imageManip_sample(image, size, x, y);
        }
    }

    for (i = 0; i < newSize * newSize; i++) {
        newImage[i] = acc[i] / (upscale * upscale);
    }

    free(upscaled);
    free(acc);

    return newImage;
}

uint8_t * imageManip_skew(uint8_t * image, size_t size,
    float m11, float m12, float m21, float m22, float tx, float ty) {

    uint8_t * newImage = (uint8_t *)malloc(size * size);

    float scale = (float)2 / (float)size;

    int i;
    int j;
    for (j = 0; j < size; j++) {
        float y = ((float)j + 0.5) * scale - 1;
        for (i = 0; i < size; i++) {
            float x = ((float)i + 0.5) * scale - 1;

            float xp = m11*x + m12*y + tx;
            float yp = m21*x + m22*y + ty;

            newImage[size*j + i] = imageManip_sample(image, size, xp, yp);
        }
    }

    return newImage;
}

float imageManip_floatRand() {
    return (float)rand() / RAND_MAX;
}

void downscaleImages(int size) {
    unsigned int nImages, nLabels, width, height;
    uint8_t ** images = dataset_loadFeatures("mnist/train-images.idx3-ubyte",
        &nImages, &width, &height);
    uint8_t * labels = dataset_loadLabels("mnist/train-labels.idx1-ubyte",
        &nLabels);

    uint8_t  ** newImages = malloc(sizeof(uint8_t*) * nImages);

    int i;
    for (i = 0; i < nImages; i++) {
        newImages[i] = imageManip_downsize(images[i], width, size);
        printf("\r%d/%d      ", i+1, nImages); fflush(stdout);
    }

    char imagesFn[1000];
    char labelsFn[1000];
    sprintf(imagesFn, "mnist/train-images-%dx%d-60000.idx3-ubyte", size, size);
    sprintf(labelsFn, "mnist/train-labels-%dx%d-60000.idx1-ubyte", size, size);

    dataset_saveFeatures(imagesFn, newImages, nImages, size, size);
    dataset_saveLabels(labelsFn, labels, nLabels);

    dataset_deleteFeatures(images, nImages);
    dataset_deleteFeatures(newImages, nImages);
    free(labels);
}

void downscaleTestImages(int size) {
    unsigned int nImages, nLabels, width, height;
    uint8_t ** images = dataset_loadFeatures("mnist/t10k-images.idx3-ubyte",
        &nImages, &width, &height);
    uint8_t * labels = dataset_loadLabels("mnist/t10k-labels.idx1-ubyte",
        &nLabels);

    uint8_t  ** newImages = malloc(sizeof(uint8_t*) * nImages);

    int i;
    for (i = 0; i < nImages; i++) {
        newImages[i] = imageManip_downsize(images[i], width, size);
        printf("\r%d/%d      ", i+1, nImages); fflush(stdout);
    }

    char imagesFn[1000];
    char labelsFn[1000];
    sprintf(imagesFn, "mnist/test-images-%dx%d-60000.idx3-ubyte", size, size);
    sprintf(labelsFn, "mnist/test-labels-%dx%d-10000.idx1-ubyte", size, size);

    dataset_saveFeatures(imagesFn, newImages, nImages, size, size);
    dataset_saveLabels(labelsFn, labels, nLabels);

    dataset_deleteFeatures(images, nImages);
    dataset_deleteFeatures(newImages, nImages);
    free(labels);
}

int main() {
    unsigned int nImages, nLabels, width, height;
    uint8_t ** images = dataset_loadFeatures("mnist/train-images.idx3-ubyte",
        &nImages, &width, &height);
    uint8_t * labels = dataset_loadLabels("mnist/train-labels.idx1-ubyte",
        &nLabels);

    for (int i = 27; i >= 9; i--) {
        printf("\nDownscaling train images %d\n", i);
        downscaleImages(i);
        printf("\nDownscaling test images %d\n", i);
        downscaleTestImages(i);
    }

    return 0;
}
