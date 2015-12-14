#ifndef MNIST_H
#define MNIST_H

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

using namespace std;
using namespace caffe;
typedef double Dtype;

class MNIST
{
public:
    MNIST(string path);

private:
    uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }

public:
    uint32_t rows;
    uint32_t cols;
    uint32_t num_train_images, num_test_images;
    uint32_t num_train_labels, num_test_labels;

    Blob<Dtype>* blob_train_images;
    Blob<Dtype>* blob_test_images;
    Blob<Dtype>* blob_train_labels;
    Blob<Dtype>* blob_test_labels;

};

#endif // MNIST_H
