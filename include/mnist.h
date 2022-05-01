
#ifndef HDC_MNIST_H
#define HDC_MNIST_H

unsigned char * mnist_loadLabels(const char * filePath, unsigned int * nItems);
void mnist_saveLabels(const char * filePath, unsigned char * labels, unsigned int nLabels);

void mnist_deleteImages(unsigned char ** images, unsigned int n);

unsigned char ** mnist_loadImages(const char * filePath, unsigned int * nItems,
    unsigned int * width, unsigned int * height);

void mnist_saveImages(const char * filePath, unsigned char ** images, 
    unsigned int nImages, unsigned int width, unsigned int height);

unsigned char ** mnist_resizeImages(unsigned char ** images, unsigned int nImages,
    unsigned int * widthPtr, unsigned int * heightPtr, unsigned int downscale);

#endif // HDC_MNIST_H