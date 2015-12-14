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

int nIter = 100;
int clas = 10; //The number of classes in MNIST dataset


int main()
{
    MNIST data("/home/koosy/caffe/caffe/data/mnist/");

//    for (int n=0;n<data.num_test_images;n++){
//        cout<<data.blob_test_labels->mutable_cpu_data()[data.blob_test_labels->offset(n,0,0.0)]<<": " << endl;
//        for(int r=0;r<data.rows;r++){
//            for(int c=0; c<data.cols;c++){
//                //raw data
//                int px = data.blob_test_images->mutable_cpu_data()[data.blob_test_images->offset(n,0,r,c)];
//                cout << setfill(' ') << setw(4) << px;

//            }
//            cout << endl;
//        }
//        cout<<endl;
//    }


    //set inner product layer
    vector<Blob<Dtype>*> blob_bottom_ip_vec_;
    vector<Blob<Dtype>*> blob_top_ip_vec_;
    Blob<Dtype>* const blob_top_ip_ = new Blob<Dtype>();

    blob_bottom_ip_vec_.push_back(data.blob_train_images);
    blob_top_ip_vec_.push_back(blob_top_ip_);

    LayerParameter layer_ip_param;
    layer_ip_param.mutable_inner_product_param()->set_num_output(10);
    layer_ip_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
    layer_ip_param.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");

    InnerProductLayer<Dtype> layer_ip(layer_ip_param);
    layer_ip.SetUp(blob_bottom_ip_vec_, blob_top_ip_vec_);

    // set softmax loss layer
    vector<Blob<Dtype>*> blob_bottom_loss_vec_;
    vector<Blob<Dtype>*> blob_top_loss_vec_;
    Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

    blob_bottom_loss_vec_.push_back(blob_top_ip_);
    blob_bottom_loss_vec_.push_back(data.blob_train_labels);
    blob_top_loss_vec_.push_back(blob_top_loss_);

    LayerParameter layer_loss_param;
    SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
    layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);\


    // forward and backward iteration
    for(int n=0;n<nIter;n++){
        // forward
        layer_ip.Forward(blob_bottom_ip_vec_, blob_top_ip_vec_);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_vec_, blob_top_loss_vec_);
        cout<<"Iter "<<n<<" loss "<<loss<<endl;

        // backward
        vector<bool> backpro_vec;
        backpro_vec.push_back(1);
        backpro_vec.push_back(0);
        layer_loss.Backward(blob_top_loss_vec_, backpro_vec, blob_bottom_loss_vec_);
        layer_ip.Backward(blob_top_ip_vec_, backpro_vec, blob_bottom_ip_vec_);

        // update weights of layer_ip
        Dtype rate = 0.1;
        vector<shared_ptr<Blob<Dtype> > > param = layer_ip.blobs();
        caffe_scal(param[0]->count(), rate, param[0]->mutable_cpu_diff());
        param[0]->Update();

        // show weight params and derv of the ip layer
//        for(int i=0;i<param[0]->count();i++){
         //  cout<<i<<": "<<param[0]->cpu_data()[i]<<", diff: "<<param[0]->mutable_cpu_diff()[i]<<endl;
        //}
        //cout<<endl;
    }


    //prediction
    vector<Blob<Dtype>*> blob_bottom_ip_test_vec_;
    vector<Blob<Dtype>*> blob_top_ip_test_vec_;
    Blob<Dtype>* const blob_top_ip_test_ = new Blob<Dtype>();

    blob_bottom_ip_test_vec_.push_back(data.blob_test_images);
    blob_top_ip_test_vec_.push_back(blob_top_ip_test_);

    layer_ip.Reshape(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);
    layer_ip.Forward(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);

    //evaluation
    //score
    int correct =0;
    for (int n = 0; n<data.num_test_images;n++){  // 10 <<-- num_test
        int truelabel = data.blob_test_labels->mutable_cpu_data()[data.blob_test_labels->offset(n,0,0,0)];

        double* score = new double[10];
        for(int i = 0 ; i<10;i++){
            score[i]=blob_top_ip_test_-> mutable_cpu_data()[blob_top_ip_test_->offset(n,i,0,0)];
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


        cout << "True label: " << truelabel << "      Predicted label:" << predictedlabel << "   Accuracy: " << correct <<"/" << (n+1) <<endl;
        cout << score[0] << " " << score[1] << " " << score[2] << " " << score[3] << " " << score[4] << " ";
        cout << score[5] << " " << score[6] << " " << score[7] << " " << score[8] << " " << score[9] << endl;


    }
    return 0;
}

