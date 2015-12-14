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

int main(int argc, char** argv) {
    MNIST data("/home/VI/staff/koosy/caffe/caffe-rc2/data/mnist/");

    int n = 0;
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            int px = data.blob_train_images->cpu_data()[data.blob_train_images->offset(n,0,i,j)];
            cout << setfill(' ') << setw(4) << px;
        }
        cout<<endl;
    }
    int label = data.blob_train_labels->cpu_data()[data.blob_train_labels->offset(n,0,0,0)];
    cout<<"label: "<<label<<" "<<endl;

    return 0;
}


