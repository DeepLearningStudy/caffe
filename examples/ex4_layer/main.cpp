#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;

typedef double Dtype;

int main(int argc, char** argv) {
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_random_seed(1701);

    // bottom/top blobs
    Blob<Dtype>* blob_top = new Blob<Dtype>();
    Blob<Dtype>* blob_bottom = new Blob<Dtype>(10, 20, 1, 1);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom);

    // blob vector
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    blob_bottom_vec.push_back(blob_bottom);
    blob_top_vec.push_back(blob_top);

    // argmax layer
    LayerParameter layer_param;
    ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
    argmax_param->set_out_max_val(true);    // two channel: 0 arg, 1 max_val
    ArgMaxLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);

    cout<<"blob_top_num:"<<blob_top->num()<<endl;
    cout<<"blob_bottom_num:"<<blob_bottom->num()<<endl;
    cout<<"blob_top_channels:"<<blob_top->channels()<<endl;

    //forward
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Now, check values
    const Dtype* bottom_data = blob_bottom->cpu_data();
    const Dtype* top_data = blob_top->cpu_data();

    int max_ind;
    Dtype max_val;
    int num = blob_bottom->num();
    int dim = blob_bottom->count() / num;
    for (int i = 0; i < num; ++i) {
        max_ind = top_data[blob_top->offset(i,0,0,0)];
        max_val = top_data[blob_top->offset(i,1,0,0)];
        cout<<"max_ind:"<<max_ind<<endl;
        cout<<"max_val:"<<max_val<<endl;
        for (int j = 0; j < dim; ++j) {
            cout<<bottom_data[i * dim + j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}


