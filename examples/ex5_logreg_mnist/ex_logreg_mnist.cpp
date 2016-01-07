#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

typedef double Dtype;
using namespace caffe;
using namespace std;

int nIter = 10000;
int clas = 10; //The number of classes in MNIST dataset


int main()
{
    Caffe::set_mode(Caffe::GPU);

    // data layer
    vector<Blob<Dtype>*> blob_bottom_data_vec;
    vector<Blob<Dtype>*> blob_top_data_vec;
    Blob<Dtype>* const blob_data = new Blob<Dtype>();
    Blob<Dtype>* const blob_label = new Blob<Dtype>();

    blob_top_data_vec.push_back(blob_data);
    blob_top_data_vec.push_back(blob_label);

    LayerParameter layer_data_param;
    DataParameter* data_param = layer_data_param.mutable_data_param();
    data_param->set_batch_size(64);
    data_param->set_source("/home/koosy/caffe/caffe/examples/mnist/mnist_train_lmdb");
    data_param->set_backend(caffe::DataParameter_DB_LMDB);

    TransformationParameter* transform_param = layer_data_param.mutable_transform_param();
    transform_param->set_scale(1./255.);

    DataLayer<Dtype> layer_data(layer_data_param);
    layer_data.SetUp(blob_bottom_data_vec, blob_top_data_vec);

    //set inner product layer
    vector<Blob<Dtype>*> blob_bottom_ip_vec;
    vector<Blob<Dtype>*> blob_top_ip_vec;
    Blob<Dtype>* const blob_top_ip = new Blob<Dtype>();

    blob_bottom_ip_vec.push_back(blob_data);
    blob_top_ip_vec.push_back(blob_top_ip);

    LayerParameter layer_ip_param;
    layer_ip_param.mutable_inner_product_param()->set_num_output(10);
    layer_ip_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
    layer_ip_param.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");

    InnerProductLayer<Dtype> layer_ip(layer_ip_param);
    layer_ip.SetUp(blob_bottom_ip_vec, blob_top_ip_vec);

    // set softmax loss layer
    vector<Blob<Dtype>*> blob_bottom_loss_vec;
    vector<Blob<Dtype>*> blob_top_loss_vec;
    Blob<Dtype>* const blob_top_loss = new Blob<Dtype>();

    blob_bottom_loss_vec.push_back(blob_top_ip);
    blob_bottom_loss_vec.push_back(blob_label);
    blob_top_loss_vec.push_back(blob_top_loss);

    LayerParameter layer_loss_param;
    SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
    layer_loss.SetUp(blob_bottom_loss_vec, blob_top_loss_vec);\


    // forward and backward iteration
    for(int n=0;n<nIter;n++){
        // forward
        layer_data.Forward(blob_bottom_data_vec, blob_top_data_vec);
        layer_ip.Forward(blob_bottom_ip_vec, blob_top_ip_vec);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_vec, blob_top_loss_vec);
        cout<<"Iter "<<n<<" loss "<<loss<<endl;

        // backward
        vector<bool> backpro_vec;
        backpro_vec.push_back(1);
        backpro_vec.push_back(0);
        layer_loss.Backward(blob_top_loss_vec, backpro_vec, blob_bottom_loss_vec);
        layer_ip.Backward(blob_top_ip_vec, backpro_vec, blob_bottom_ip_vec);

        // update weights of layer_ip
        Dtype rate = 0.1;
        vector<shared_ptr<Blob<Dtype> > > param = layer_ip.blobs();
        caffe_scal(param[0]->count(), rate, param[0]->mutable_cpu_diff());
        param[0]->Update();
    }


    //prediction
    // data layer
    vector<Blob<Dtype>*> blob_bottom_testdata_vec;
    vector<Blob<Dtype>*> blob_top_testdata_vec;
    Blob<Dtype>* const blob_testdata = new Blob<Dtype>();
    Blob<Dtype>* const blob_testlabel = new Blob<Dtype>();

    blob_top_testdata_vec.push_back(blob_testdata);
    blob_top_testdata_vec.push_back(blob_testlabel);

    LayerParameter layer_testdata_param;
    DataParameter* testdata_param = layer_testdata_param.mutable_data_param();
    testdata_param->set_batch_size(10000);
    testdata_param->set_source("/home/koosy/caffe/caffe/examples/mnist/mnist_test_lmdb");
    testdata_param->set_backend(caffe::DataParameter_DB_LMDB);

    TransformationParameter* transform_test_param = layer_testdata_param.mutable_transform_param();
    transform_test_param->set_scale(1./255.);

    DataLayer<Dtype> layer_testdata(layer_testdata_param);
    layer_testdata.SetUp(blob_bottom_testdata_vec, blob_top_testdata_vec);

    vector<Blob<Dtype>*> blob_bottom_ip_test_vec;
    blob_bottom_ip_test_vec.push_back(blob_testdata);

    layer_ip.Reshape(blob_bottom_ip_test_vec, blob_top_ip_vec);

    vector<Blob<Dtype>*> blob_bottom_loss_test_vec;
    blob_bottom_loss_test_vec.push_back(blob_top_ip);
    blob_bottom_loss_test_vec.push_back(blob_testlabel);

    layer_loss.Reshape(blob_bottom_loss_test_vec, blob_top_loss_vec);

    // armax layer
    vector<Blob<Dtype>*> blob_bottom_argmax_vec;
    vector<Blob<Dtype>*> blob_top_argmax_vec;
    Blob<Dtype>* blob_top_argmax = new Blob<Dtype>();
    blob_bottom_argmax_vec.push_back(blob_top_ip);
    blob_top_argmax_vec.push_back(blob_top_argmax);

    LayerParameter layer_argmax_param;
    ArgMaxParameter* argmax_param = layer_argmax_param.mutable_argmax_param();
    argmax_param->set_out_max_val(false);
    ArgMaxLayer<Dtype> layer_argmax(layer_argmax_param);
    layer_argmax.SetUp(blob_bottom_argmax_vec, blob_top_argmax_vec);

    //evaluation
    int correct = 0;
    int cnt = 0;
    // forward
    layer_testdata.Forward(blob_bottom_testdata_vec, blob_top_testdata_vec);
    layer_ip.Forward(blob_bottom_ip_test_vec, blob_top_ip_vec);
    layer_argmax.Forward(blob_bottom_argmax_vec, blob_top_argmax_vec);

    Dtype loss = layer_loss.Forward(blob_bottom_loss_test_vec, blob_top_loss_vec);
    cout<<" loss "<<loss<<endl;

    for (int n = 0; n<blob_testlabel->count();n++){  // 10 <<-- num_test
        cnt ++;
        Dtype* label_data = blob_testlabel->mutable_cpu_data();
        int truelabel = label_data[n];

        Dtype* prediction_data = blob_top_argmax-> mutable_cpu_data();
        int predictedlabel = prediction_data[n];

        if(truelabel == predictedlabel){
            correct++;
        }

        cout << "True label: " << truelabel << "      Predicted label:" << predictedlabel << "   Accuracy: " << correct <<"/" << cnt <<endl;
    }

    return 0;
}

