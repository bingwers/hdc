
#ifndef HDC_DATASET_H
#define HDC_DATASET_H

#include <stdint.h>

typedef struct Dataset Dataset;

struct Dataset {
    unsigned int nItems;
    unsigned int width, height;
    uint8_t * labels;
    uint8_t ** features;
};

uint8_t * dataset_loadLabels(const char * filePath, uint32_t * nItems);
void dataset_saveLabels(const char * filePath, uint8_t * labels, uint32_t nLabels);

void dataset_deleteFeatures(uint8_t ** features, uint32_t n);

uint8_t ** dataset_loadFeatures(const char * filePath, uint32_t * nItems,
    uint32_t * width, uint32_t * height);

void dataset_saveFeatures(const char * filePath, uint8_t ** features, 
    uint32_t nItems, uint32_t width, uint32_t height);

Dataset * Dataset_load(const char * labelsFn,
    const char * featuresFn, size_t downscale);

void Dataset_delete(Dataset * dataset);

#endif // HDC_DATASET_H