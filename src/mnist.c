
#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

unsigned int flipByteOrder32(unsigned int val) {
    return ((val & 0xFF) << 24)
            + ((val & 0xFF00) << 8)
            + ((val & 0xFF0000) >> 8)
            + ((val & 0xFF000000) >> 24);
}

unsigned char * mnist_loadLabels(const char * filePath, unsigned int * nItems) {
    FILE * fp = fopen(filePath, "r");
    if (fp == NULL) {
        return NULL;
    }

    unsigned int magicNumber;

    if (fread(&magicNumber, 4, 1, fp) != 1
        || fread(nItems, 4, 1, fp) != 1) {
        
        fclose(fp);
        return NULL;
    }

    *nItems = flipByteOrder32(*nItems);

    unsigned char * labels = (char*)malloc(*nItems * sizeof(char));

    if (fread(labels, 1, *nItems, fp) != *nItems) {
        fclose(fp);
        free(labels);
        return NULL;
    }

    fclose(fp);

    return labels;
}

void mnist_saveLabels(const char * filePath, unsigned char * labels, unsigned int nLabels) {
    FILE * fp = fopen(filePath, "wb");

    unsigned int magicNumber = 0x08010000; //  might be wrong
    unsigned int nLabels_flipped = flipByteOrder32(nLabels);

    fwrite(&magicNumber, 4, 1, fp);
    fwrite(&nLabels_flipped, 4, 1, fp);

    fwrite(labels, 1, nLabels, fp);

    fclose(fp);
}

void mnist_deleteImages(unsigned char ** images, unsigned int n) {
    unsigned int i;
    for (i = 0; i < n; ++i) {
        free(images[i]);
    }

    free(images);
}

unsigned char ** mnist_loadImages(const char * filePath, unsigned int * nItems,
    unsigned int * width, unsigned int * height) {

    FILE * fp = fopen(filePath, "r");

    if (fp == NULL) {
        return NULL;
    }

    unsigned int magicNumber;
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

    unsigned char ** images = (unsigned char**)malloc(sizeof(unsigned char **) * (*nItems));
    unsigned int imageSize = (*width) * (*height);

    unsigned int i;
    for (i = 0; i < *nItems; ++i) {
        unsigned char * image = (unsigned char*)malloc(sizeof(unsigned char) * imageSize);
        images[i] = image;

        if (fread(image, 1, imageSize, fp) != imageSize) {
            mnist_deleteImages(images, i);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);

    return images;
}

void mnist_saveImages(const char * filePath, unsigned char ** images, 
    unsigned int nImages, unsigned int width, unsigned int height) {

    FILE * fp = fopen(filePath, "wb");

    unsigned int magicNumber = 0x08030000; // might be wrong
    unsigned int nImages_flipped = flipByteOrder32(nImages);
    unsigned int width_flipped = flipByteOrder32(width);
    unsigned int height_flipped = flipByteOrder32(height);

    unsigned int imageSize = width * height;

    fwrite(&magicNumber, 4, 1, fp);
    fwrite(&nImages_flipped, 4, 1, fp);
    fwrite(&width_flipped, 4, 1, fp);
    fwrite(&height_flipped, 4, 1, fp);

    int i;
    for (i = 0; i < nImages; i++) {
        fwrite(images[i], 1, imageSize, fp);
    }

    fclose(fp);
}

unsigned char ** mnist_resizeImages(unsigned char ** images, unsigned int nImages,
    unsigned int * widthPtr, unsigned int * heightPtr, unsigned int downscale) {
    
    unsigned int width = *widthPtr;
    unsigned int height = *heightPtr;

    unsigned int newWidth = width / downscale;
    unsigned int newHeight = height / downscale;
    unsigned int newImageSize = newWidth * newHeight;

    unsigned char ** newImages = (unsigned char**)malloc(sizeof(unsigned char **) * (nImages));

    int i; for (i = 0; i < nImages; i++) {
        unsigned char * image = images[i];
        unsigned char * newImage = (unsigned char*)malloc(sizeof(unsigned char) * newImageSize);

        int j; for (j = 0; j < newHeight; j++) {
            int k; for (k = 0; k < newWidth; k++) {
                int val = 0;
                int l; for (l = 0; l < downscale; l++) {
                    int m; for (m = 0; m < downscale; m++) {
                        val += (int)image[(downscale * j + l) * width + downscale * k + m];
                    }
                }

                val /= (downscale * downscale);
                newImage[j * newWidth + k] = (unsigned char)val;
            }
        }
        
        newImages[i] = newImage;
    }

    *widthPtr = newWidth;
    *heightPtr = newHeight;

    return newImages;
}
