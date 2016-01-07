#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "mnist.h"

using namespace caffe;
using namespace std;
using namespace cv;

int nIter = 1000;
int clas = 10; //The number of classes in MNIST dataset


int main()
{
    Caffe::set_mode(Caffe::GPU);

    /// LeNet from lenet_auto_train.prototxt
    // data layer
    vector<Blob<Dtype>*> blob_bottom_data_vec_;
    vector<Blob<Dtype>*> blob_top_data_vec_;
    Blob<Dtype>* const blob_data = new Blob<Dtype>();
    Blob<Dtype>* const blob_label = new Blob<Dtype>();

    blob_top_data_vec_.push_back(blob_data);
    blob_top_data_vec_.push_back(blob_label);

    LayerParameter layer_data_param;
    DataParameter* data_param = layer_data_param.mutable_data_param();
    data_param->set_batch_size(64);
    data_param->set_source("/home/koosy/caffe/caffe/examples/mnist/mnist_train_lmdb");
    data_param->set_backend(caffe::DataParameter_DB_LMDB);

    TransformationParameter* transform_param = layer_data_param.mutable_transform_param();
    transform_param->set_scale(1./255.);

    DataLayer<Dtype> layer_data(layer_data_param);
    layer_data.SetUp(blob_bottom_data_vec_, blob_top_data_vec_);

    // conv1
    vector<Blob<Dtype>*> blob_bottom_conv1_vec_;
    vector<Blob<Dtype>*> blob_top_conv1_vec_;
    Blob<Dtype>* const blob_top_conv1_ = new Blob<Dtype>();

    blob_bottom_conv1_vec_.push_back(blob_data);
    blob_top_conv1_vec_.push_back(blob_top_conv1_);

    LayerParameter layer_conv1_param;
    ConvolutionParameter* conv1_param = layer_conv1_param.mutable_convolution_param();
    conv1_param->set_num_output(20);
    conv1_param->add_kernel_size(5);
    conv1_param->mutable_weight_filler()->set_type("xavier");

    ConvolutionLayer<Dtype> layer_conv1(layer_conv1_param);
    layer_conv1.SetUp(blob_bottom_conv1_vec_, blob_top_conv1_vec_);

    // pool1
    vector<Blob<Dtype>*> blob_bottom_pool1_vec_;
    vector<Blob<Dtype>*> blob_top_pool1_vec_;
    Blob<Dtype>* const blob_top_pool1_ = new Blob<Dtype>();

    blob_bottom_pool1_vec_.push_back(blob_top_conv1_);
    blob_top_pool1_vec_.push_back(blob_top_pool1_);

    LayerParameter layer_pool1_param;
    PoolingParameter* pool1_param = layer_pool1_param.mutable_pooling_param();
    pool1_param->set_pool(caffe::PoolingParameter::MAX);
    pool1_param->set_kernel_size(2);
    pool1_param->set_stride(2);

    PoolingLayer<Dtype> layer_pool1(layer_pool1_param);
    layer_pool1.SetUp(blob_bottom_pool1_vec_, blob_top_pool1_vec_);

    // conv2
    vector<Blob<Dtype>*> blob_bottom_conv2_vec_;
    vector<Blob<Dtype>*> blob_top_conv2_vec_;
    Blob<Dtype>* const blob_top_conv2_ = new Blob<Dtype>();

    blob_bottom_conv2_vec_.push_back(blob_top_pool1_);
    blob_top_conv2_vec_.push_back(blob_top_conv2_);

    LayerParameter layer_conv2_param;
    ConvolutionParameter* conv2_param = layer_conv2_param.mutable_convolution_param();
    conv2_param->set_num_output(50);
    conv2_param->add_kernel_size(5);
    conv2_param->mutable_weight_filler()->set_type("xavier");

    ConvolutionLayer<Dtype> layer_conv2(layer_conv2_param);
    layer_conv2.SetUp(blob_bottom_conv2_vec_, blob_top_conv2_vec_);

    // pool2
    vector<Blob<Dtype>*> blob_bottom_pool2_vec_;
    vector<Blob<Dtype>*> blob_top_pool2_vec_;
    Blob<Dtype>* const blob_top_pool2_ = new Blob<Dtype>();

    blob_bottom_pool2_vec_.push_back(blob_top_conv2_);
    blob_top_pool2_vec_.push_back(blob_top_pool2_);

    LayerParameter layer_pool2_param;
    PoolingParameter* pool2_param = layer_pool2_param.mutable_pooling_param();
    pool2_param->set_pool(caffe::PoolingParameter::MAX);
    pool2_param->set_kernel_size(2);
    pool2_param->set_stride(2);

    PoolingLayer<Dtype> layer_pool2(layer_pool2_param);
    layer_pool2.SetUp(blob_bottom_pool2_vec_, blob_top_pool2_vec_);

    // ip1
    vector<Blob<Dtype>*> blob_bottom_ip1_vec_;
    vector<Blob<Dtype>*> blob_top_ip1_vec_;
    Blob<Dtype>* const blob_top_ip1_ = new Blob<Dtype>();

    blob_bottom_ip1_vec_.push_back(blob_top_pool2_);
    blob_top_ip1_vec_.push_back(blob_top_ip1_);

    LayerParameter layer_ip1_param;
    InnerProductParameter* ip1_param = layer_ip1_param.mutable_inner_product_param();
    ip1_param->set_num_output(500);
    ip1_param->mutable_weight_filler()->set_type("xavier");

    InnerProductLayer<Dtype> layer_ip1(layer_ip1_param);
    layer_ip1.SetUp(blob_bottom_ip1_vec_, blob_top_ip1_vec_);

    // relu1
    LayerParameter layer_relu1_param;
    ReLUParameter* relu1_param = layer_relu1_param.mutable_relu_param();

    ReLULayer<Dtype> layer_relu1(layer_relu1_param);
    layer_relu1.SetUp(blob_top_ip1_vec_, blob_top_ip1_vec_);

    // ip2
    vector<Blob<Dtype>*> blob_bottom_ip2_vec_;
    vector<Blob<Dtype>*> blob_top_ip2_vec_;
    Blob<Dtype>* const blob_top_ip2_ = new Blob<Dtype>();

    blob_bottom_ip2_vec_.push_back(blob_top_ip1_);
    blob_top_ip2_vec_.push_back(blob_top_ip2_);

    LayerParameter layer_ip2_param;
    InnerProductParameter* ip2_param = layer_ip2_param.mutable_inner_product_param();
    ip2_param->set_num_output(10);
    ip2_param->mutable_weight_filler()->set_type("xavier");

    InnerProductLayer<Dtype> layer_ip2(layer_ip2_param);
    layer_ip2.SetUp(blob_bottom_ip2_vec_, blob_top_ip2_vec_);

    //loss
    vector<Blob<Dtype>*> blob_bottom_loss_vec_;
    vector<Blob<Dtype>*> blob_top_loss_vec_;
    Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

    blob_bottom_loss_vec_.push_back(blob_top_ip2_);
    blob_bottom_loss_vec_.push_back(blob_label);
    blob_top_loss_vec_.push_back(blob_top_loss_);

    LayerParameter layer_loss_param;
    SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
    layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);

    // Forward
    for(int n=0;n<nIter;n++){
        cout<<"Iter "<<n;
        // forward
        layer_data.Forward(blob_bottom_data_vec_, blob_top_data_vec_);
        //        for(int i=0;i<64;i++){
        //            cout<<blob_label->mutable_cpu_data()[blob_label->offset(i,0,0,0)]<<" ";
        //        }
        //        int truelabel = blob_testlabel->mutable_cpu_data()[blob_testlabel->offset(n,0,0,0)];

        layer_conv1.Forward(blob_bottom_conv1_vec_, blob_top_conv1_vec_);
        layer_pool1.Forward(blob_bottom_pool1_vec_, blob_top_pool1_vec_);
        layer_conv2.Forward(blob_bottom_conv2_vec_, blob_top_conv2_vec_);
        layer_pool2.Forward(blob_bottom_pool2_vec_, blob_top_pool2_vec_);
        layer_ip1.Forward(blob_bottom_ip1_vec_, blob_top_ip1_vec_);
        layer_relu1.Forward(blob_top_ip1_vec_, blob_top_ip1_vec_);
        layer_ip2.Forward(blob_bottom_ip2_vec_, blob_top_ip2_vec_);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_vec_, blob_top_loss_vec_);

        cout<<" loss "<<loss<<endl;

        // backward
        vector<bool> backpro_vec1, backpro_vec2;
        backpro_vec1.push_back(1);
        backpro_vec1.push_back(0);

        backpro_vec2.push_back(1);
        layer_loss.Backward(blob_top_loss_vec_, backpro_vec1, blob_bottom_loss_vec_);
        layer_ip2.Backward(blob_top_ip2_vec_, backpro_vec2, blob_bottom_ip2_vec_);
        layer_relu1.Backward(blob_top_ip1_vec_, backpro_vec2, blob_top_ip1_vec_);
        layer_ip1.Backward(blob_top_ip1_vec_, backpro_vec2, blob_bottom_ip1_vec_);
        layer_pool2.Backward(blob_top_pool2_vec_, backpro_vec2, blob_bottom_pool2_vec_ );
        layer_conv2.Backward(blob_top_conv2_vec_, backpro_vec2, blob_bottom_conv2_vec_ );
        layer_pool1.Backward(blob_top_pool1_vec_, backpro_vec2, blob_bottom_pool1_vec_ );
        layer_conv1.Backward(blob_top_conv1_vec_, backpro_vec2, blob_bottom_conv1_vec_ );


        // update weights of layer_ip
        Dtype rate = 0.1;
        vector<shared_ptr<Blob<Dtype> > > param_conv1 = layer_conv1.blobs();
        caffe_gpu_scal(param_conv1[0]->count(), rate, param_conv1[0]->mutable_gpu_diff());
        param_conv1[0]->Update();

        vector<shared_ptr<Blob<Dtype> > > param_conv2 = layer_conv2.blobs();
        param_conv2[0]->gpu_data();
        caffe_gpu_scal(param_conv2[0]->count(), rate, param_conv2[0]->mutable_gpu_diff());
        param_conv2[0]->Update();

        vector<shared_ptr<Blob<Dtype> > > param_ip1 = layer_ip1.blobs();
        param_ip1[0]->gpu_data();
        caffe_gpu_scal(param_ip1[0]->count(), rate, param_ip1[0]->mutable_gpu_diff());
        param_ip1[0]->Update();

        vector<shared_ptr<Blob<Dtype> > > param_ip2 = layer_ip2.blobs();
        param_ip2[0]->gpu_data();
        caffe_gpu_scal(param_ip2[0]->count(), rate, param_ip2[0]->mutable_gpu_diff());
        param_ip2[0]->Update();
    }


    //prediction
    // data layer
    vector<Blob<Dtype>*> blob_bottom_testdata_vec_;
    vector<Blob<Dtype>*> blob_top_testdata_vec_;
    Blob<Dtype>* const blob_testdata = new Blob<Dtype>();
    Blob<Dtype>* const blob_testlabel = new Blob<Dtype>();

    blob_top_testdata_vec_.push_back(blob_testdata);
    blob_top_testdata_vec_.push_back(blob_testlabel);

    LayerParameter layer_testdata_param;
    DataParameter* testdata_param = layer_testdata_param.mutable_data_param();
    testdata_param->set_batch_size(10000);
    testdata_param->set_source("/home/koosy/caffe/caffe/examples/mnist/mnist_test_lmdb");
    testdata_param->set_backend(caffe::DataParameter_DB_LMDB);

    TransformationParameter* transform_test_param = layer_testdata_param.mutable_transform_param();
    transform_test_param->set_scale(1./255.);

    DataLayer<Dtype> layer_testdata(layer_testdata_param);
    layer_testdata.SetUp(blob_bottom_testdata_vec_, blob_top_testdata_vec_);

    // reshape
    vector<Blob<Dtype>*> blob_bottom_conv1_test_vec_;
    blob_bottom_conv1_test_vec_.push_back(blob_testdata);
    layer_conv1.Reshape(blob_bottom_conv1_test_vec_, blob_top_conv1_vec_);
    layer_pool1.Reshape(blob_bottom_pool1_vec_, blob_top_pool1_vec_);
    layer_conv2.Reshape(blob_bottom_conv2_vec_, blob_top_conv2_vec_);
    layer_pool2.Reshape(blob_bottom_pool2_vec_, blob_top_pool2_vec_);
    layer_ip1.Reshape(blob_bottom_ip1_vec_, blob_top_ip1_vec_);
    layer_relu1.Reshape(blob_top_ip1_vec_, blob_top_ip1_vec_);
    layer_ip2.Reshape(blob_bottom_ip2_vec_, blob_top_ip2_vec_);

    vector<Blob<Dtype>*> blob_bottom_loss_test_vec_;
    blob_bottom_loss_test_vec_.push_back(blob_top_ip2_);
    blob_bottom_loss_test_vec_.push_back(blob_testlabel);

    layer_loss.Reshape(blob_bottom_loss_test_vec_, blob_top_loss_vec_);

    // evaluation on 100 iteration
    int correct = 0;
    int cnt = 0;
    for(int k = 0;k<1;k++){
        // forward
        layer_testdata.Forward(blob_bottom_testdata_vec_, blob_top_testdata_vec_);
        layer_conv1.Forward(blob_bottom_conv1_test_vec_, blob_top_conv1_vec_);
        layer_pool1.Forward(blob_bottom_pool1_vec_, blob_top_pool1_vec_);
        layer_conv2.Forward(blob_bottom_conv2_vec_, blob_top_conv2_vec_);
        layer_pool2.Forward(blob_bottom_pool2_vec_, blob_top_pool2_vec_);
        layer_ip1.Forward(blob_bottom_ip1_vec_, blob_top_ip1_vec_);
        layer_relu1.Forward(blob_top_ip1_vec_, blob_top_ip1_vec_);
        layer_ip2.Forward(blob_bottom_ip2_vec_, blob_top_ip2_vec_);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_test_vec_, blob_top_loss_vec_);
        cout<<" loss "<<loss<<endl;

        //evaluation
        //score
        for (int n = 0; n<blob_testlabel->count();n++){  // 10 <<-- num_test
            cnt ++;
            Dtype* label_data = blob_testlabel->mutable_cpu_data();
            int truelabel = label_data[n];

            Dtype* score_data = blob_top_ip2_-> mutable_cpu_data();
            double* score = new double[10];
            for(int i = 0 ; i<10;i++){
                score[i]=score_data[blob_top_ip2_->offset(n,i,0,0)];
            }

            int predictedlabel =0;

            for(int i = 1;i<10;i++){
                if(score[i]>score[predictedlabel]){
                    predictedlabel = i;
                }
            }

            if(truelabel == predictedlabel){
                correct++;
            }

            cout << "True label: " << truelabel << "      Predicted label:" << predictedlabel << "   Accuracy: " << correct <<"/" << cnt <<endl;

        }
    }
    return 0;
}

