#include "mnist.h"

MNIST::MNIST(string path)
{
    //Read training data

    string filename_train_image = path + "train-images-idx3-ubyte";
    string filename_train_label = path + "train-labels-idx1-ubyte";
    string filename_test_image = path + "t10k-images-idx3-ubyte";
    string filename_test_label = path + "t10k-labels-idx1-ubyte";

    // Open files
    ifstream file_train_image(filename_train_image.c_str(), ios::in | ios::binary);
    ifstream file_train_label(filename_train_label.c_str(), ios::in | ios::binary);

    ifstream file_test_image(filename_test_image.c_str(), ios::in | ios::binary);
    ifstream file_test_label(filename_test_label.c_str(), ios::in | ios::binary);

    CHECK(file_train_image) << "Unable to open file " << filename_train_image;
    CHECK(file_train_label) << "Unable to open file " << filename_train_label;
    CHECK(file_test_image) << "Unable to open file " << filename_test_image;
    CHECK(file_test_label) << "Unable to open file " << filename_test_label;

    // Read the magic and the meta data
    uint32_t magic;

    file_train_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";

    file_train_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";

    file_test_image.seekg(4,file_test_image.beg);
    file_test_label.seekg(4,file_test_label.beg);

    file_train_image.read(reinterpret_cast<char*>(&num_train_images), 4);
    num_train_images = swap_endian(num_train_images);
    file_train_label.read(reinterpret_cast<char*>(&num_train_labels), 4);
    num_train_labels = swap_endian(num_train_labels);

    file_test_image.read(reinterpret_cast<char*>(&num_test_images), 4);
    num_test_images = swap_endian(num_test_images);
    file_test_label.read(reinterpret_cast<char*>(&num_test_labels), 4);
    num_test_labels = swap_endian(num_test_labels);


    int dim;

    CHECK_EQ(num_train_images, num_train_labels);
    file_train_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    file_train_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    file_test_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    file_test_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);


    dim = cols*rows;

    cout << "A total of " << num_train_images << " images." << endl;
    cout << "Rows: " << rows << " Cols: " << cols << endl;    

    /// training dataset
    blob_train_images = new Blob<Dtype>(num_train_images, 1, rows, cols);    
    blob_train_labels = new Blob<Dtype>(num_train_labels, 1, 1, 1);
    cout << "Training samples "<< num_train_images<< endl;

    char* train_labels = new char[num_train_labels];
    double* train_labels_d = new double[num_train_labels];
    char* train_images = new char[num_train_labels * dim];
    double* train_images_d = new double[num_train_images * dim];

    file_train_image.read(train_images, num_train_images * dim);
    for (int i = 0; i<num_train_images * dim; i++){
        train_images_d[i]=(unsigned char)train_images[i];
    }
    blob_train_images->set_cpu_data(train_images_d);

    file_train_label.read(train_labels, num_train_labels);
    for (int i = 0; i<num_train_labels; i++){
        train_labels_d[i]=(unsigned char)train_labels[i];
    }
    blob_train_labels->set_cpu_data(train_labels_d);

    /// test dataset
    blob_test_images = new Blob<Dtype>(num_test_images, 1, rows, cols);
    blob_test_labels = new Blob<Dtype>(num_test_labels, 1, 1, 1);
    cout << "Test samples "<< num_test_images<< endl;

    char* test_labels = new char[num_test_labels];
    double* test_labels_d = new double[num_test_labels];
    char* test_images = new char[num_test_labels * dim];
    double* test_images_d = new double[num_test_images * dim];

    file_test_image.read(test_images, num_test_images * dim);
    for (int i = 0; i<num_test_images * dim; i++){
        test_images_d[i]=(unsigned char)test_images[i];
    }
    blob_test_images->set_cpu_data(test_images_d);

    file_test_label.read(test_labels, num_test_labels);
    for (int i = 0; i<num_test_labels; i++){
        test_labels_d[i]=(unsigned char)test_labels[i];
    }
    blob_test_labels->set_cpu_data(test_labels_d);

    cout << "reading data done" << endl;
}

