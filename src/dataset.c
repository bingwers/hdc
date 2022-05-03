
#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"

uint32_t flipByteOrder32(uint32_t val) {
    return ((val & 0xFF) << 24)
            + ((val & 0xFF00) << 8)
            + ((val & 0xFF0000) >> 8)
            + ((val & 0xFF000000) >> 24);
}

uint8_t * dataset_loadLabels(const char * filePath, uint32_t * nItems) {
    FILE * fp = fopen(filePath, "r");
    if (fp == NULL) {
        return NULL;
    }

    uint32_t magicNumber;

    if (fread(&magicNumber, 4, 1, fp) != 1
        || fread(nItems, 4, 1, fp) != 1) {
        
        fclose(fp);
        return NULL;
    }

    *nItems = flipByteOrder32(*nItems);

    uint8_t * labels = (char*)malloc(*nItems * sizeof(uint8_t));

    if (fread(labels, 1, *nItems, fp) != *nItems) {
        fclose(fp);
        free(labels);
        return NULL;
    }

    fclose(fp);

    return labels;
}

void dataset_saveLabels(const char * filePath, uint8_t * labels, uint32_t nLabels) {
    FILE * fp = fopen(filePath, "wb");

    uint32_t magicNumber = 0x08010000; //  might be wrong
    uint32_t nLabels_flipped = flipByteOrder32(nLabels);

    fwrite(&magicNumber, 4, 1, fp);
    fwrite(&nLabels_flipped, 4, 1, fp);

    fwrite(labels, 1, nLabels, fp);

    fclose(fp);
}

void dataset_deleteFeatures(uint8_t ** features, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        free(features[i]);
    }

    free(features);
}

uint8_t ** dataset_loadFeatures(const char * filePath, uint32_t * nItems,
    uint32_t * width, uint32_t * height) {

    FILE * fp = fopen(filePath, "r");

    if (fp == NULL) {
        return NULL;
    }

    uint32_t magicNumber;
    if (fread(&magicNumber, 4, 1, fp) != 1
        || fread(nItems, 4, 1, fp) != 1
        || fread(width, 4, 1, fp) != 1
        || fread(height, 4, 1, fp) != 1) {
        
        fclose(fp);
        return NULL;
    }

    *nItems = flipByteOrder32(*nItems);
    *width = flipByteOrder32(*width);
    *height = flipByteOrder32(*height);

    uint8_t ** features = (uint8_t**)malloc(sizeof(uint8_t **) * (*nItems));
    uint32_t featureSize = (*width) * (*height);

    uint32_t i;
    for (i = 0; i < *nItems; ++i) {
        uint8_t * featureMap = (uint8_t*)malloc(sizeof(uint8_t) * featureSize);
        features[i] = featureMap;

        if (fread(featureMap, 1, featureSize, fp) != featureSize) {
            dataset_deleteFeatures(features, i);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);

    return features;
}

void dataset_saveFeatures(const char * filePath, uint8_t ** features, 
    uint32_t nItems, uint32_t width, uint32_t height) {

    FILE * fp = fopen(filePath, "wb");

    uint32_t magicNumber = 0x08030000; // might be wrong
    uint32_t nItems_flipped = flipByteOrder32(nItems);
    uint32_t width_flipped = flipByteOrder32(width);
    uint32_t height_flipped = flipByteOrder32(height);

    uint32_t featureSize = width * height;

    fwrite(&magicNumber, 4, 1, fp);
    fwrite(&nItems_flipped, 4, 1, fp);
    fwrite(&width_flipped, 4, 1, fp);
    fwrite(&height_flipped, 4, 1, fp);

    int i;
    for (i = 0; i < nItems; i++) {
        fwrite(features[i], 1, featureSize, fp);
    }

    fclose(fp);
}

Dataset * Dataset_load(const char * labelsFn,
    const char * featuresFn, size_t downscale) {

    Dataset * dataset = (Dataset*)malloc(sizeof(Dataset));

    dataset -> labels = dataset_loadLabels(labelsFn, &dataset -> nItems);    
    dataset -> features = dataset_loadFeatures(featuresFn, &dataset -> nItems,
        &dataset -> width, &dataset -> height);
    return dataset;
}

void Dataset_delete(Dataset * dataset) {
    dataset_deleteFeatures(dataset -> features, dataset -> nItems);
    free(dataset -> labels);
    free(dataset);
}
