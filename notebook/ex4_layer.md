
## 'layer' in Caffe

지난 강좌에서는 Caffe의 기본 데이터 구조인 blob에 대해서, 그리고 MNIST 데이터셋에 대해서 알아보았습니다. 이제는 본격적인 뉴럴넷 모델을 만들기 위한 핵심 빌딩블록인 layer에 대해 알아볼텐데요. 이번 시간에는 argmax layer 간단한 예를 통해 layer와 blob의 관계를 알아보고, 다음 시간에는 layer 구조를 이용하여 MNIST 데이터셋을 한번 학습해 보도록 하겠습니다. 

### Layer란?

caffe 모델에서 blob이 데이터 구조라면 layer는 연산자에 해당합니다. 아래 보시는 그림처럼 blob을 입력으로 받아서 다양한 연산을
수행하고 그 결과를 blob으로 저장해서 내보냅니다.

![그림](http://caffe.berkeleyvision.org/tutorial/fig/layer.jpg)

caffe는 다양한 종류의 layer를 지원하는데요, 예를 들어 convolution filter, pooling layer, inner
product layer, rectified-linear layer, sigmoid layer,

### Logistic Regression Classification

### Source code


    MNIST data 로딩
