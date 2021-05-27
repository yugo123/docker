[[_TOC_]]
- [GPU 서버에서 GPU 인식 확인 방법](#gpu------gpu---------)
  * [1. GPU 인식 확인](#1-gpu------)
    + [1. tensorflow](#1-tensorflow)
    + [2. pytorch](#2-pytorch)
  * [2. GPU 인식 관련 오류](#2-gpu---------)
    + [1. CUDA의 설치 여부](#1-cuda-------)
    + [2. CuDNN](#2-cudnn)
    + [3. tensorflow 및 torch 버전 확인](#3-tensorflow---torch------)
  * [3. cpu만 사용하는 코드에서 gpu 사용하는 코드로 변환 방법](#3-cpu------------gpu---------------)
    + [1. tensorflow-gpu](#1-tensorflow-gpu)
    + [2. pytorch](#2-pytorch-1)
- [Appendix](#appendix)


# GPU 서버에서 GPU 인식 확인 방법

## 1. GPU 인식 확인
### 1. tensorflow
                                
```
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

 ![](./GPU_인식_확인_방법/.figures/p1.png)

제대로 인식되었다면 위의 그림과 같이 'CPU:0'와 보유하고 있는 GPU의 종류가 나타납니다.
    
### 2. pytorch
- is_available()을 이용하는 방법입니다.
```
import torch 
torch.cuda.is_available()
``` 

 ![](./GPU_인식_확인_방법/.figures/p2.png =300x)

제대로 인식되었다면 위의 그림과 같이 True가 나타납니다. 만약 False가 나왔다면 아래의 오류 해결 방법을 확인하세요.

- device_count()을 이용하는 방법입니다.
```
import torch
torch.cuda.device_count()
```

 ![](./GPU_인식_확인_방법/.figures/p3.png =300x)

제대로 인식되었다면 위의 그림과 같이 기기에 인식된 gpu의 개수가 표시됩니다.

- device()을 이용하는 방법입니다.
```
import torch
torch.cuda.device(0)
```

 ![](./GPU_인식_확인_방법/.figures/p4.png =300x)
제대로 인식되었다면 위의 그림과 같이 device로 잡힌 gpu의 종류를 보여줍니다.

## 2. GPU 인식 관련 오류
torch.cuda.is_available()이 False로 나오거나 device_lib.list_local_devices()에서 GPU가 인식되지 않는 경우 다음의 유형을 확인해야 합니다.

GPU 서버를 이용하는 경우 2.1과 2.2는 이미 제대로 구성되었기 때문에 2.3부터 확인하세요!

### 1. CUDA의 설치 여부
CUDA는 그래픽카드에서 병렬처리를 할 수 있는 플랫폼 및 API 모델로 그래픽 카드를 이용하여 GPU 연산을 이용하기 위해서는 CUDA가 무조건 필요합니다.

그래픽 카드에 맞는 CUDA 버전은 다음의 사이트를 통해 확인할 수 있습니다.
https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

간단한 예시로 GPU 서버에는 RTX 6000 2개가 장착되어 있습니다. 위의 사이트에서 RTX 6000를 찾아보면 적정 Compute capability가 7.5로 나와있습니다.

![](./GPU_인식_확인_방법/.figures/p5.png =1000x)

이에 맞는 CUDA 버전을 확인해보면 아래와 같이 CUDA 10.0 ~ 10.2와 CUDA 11.0 ~ 11.4 버전을 사용할 수 있습니다.

![](./GPU_인식_확인_방법/.figures/p6.png =1000x)

CUDA는 https://developer.nvidia.com/cuda-toolkit-archive 에서 다운로드 받을 수 있고 위에서 찾은 버전에 맞는 Toolkit을 선택하면 됩니다. 이후의 설치 내용 및 확인 방법은 https://ghostweb.tistory.com/839을 참고하시면 됩니다.

![](./GPU_인식_확인_방법/.figures/p7.png =500x)

### 2. CuDNN
CuDNN은 tensorflow와 pytorch에서 딥러닝 학습을 수행하는데 도움을 주는 라이브러리로 CUDA 버전에 맞는 CuDNN toolkit이 필요합니다.

CuDNN은 https://developer.nvidia.com/rdp/cudnn-archive 에서 다운로드 받을 수 있고 다운로드를 하기 위해서는 회원가입 및 로그인이 필요합니다.

![](./GPU_인식_확인_방법/.figures/p8.png =1000x)

다운로드 받은 파일을 압축풀고 나면 아래와 같은 파일들이 생성되는데 이를 CUDA가 설치된 폴더에 복사하여 덮어쓰기 합니다.

![](./GPU_인식_확인_방법/.figures/p9.png =1000x)

![](./GPU_인식_확인_방법/.figures/p10.png =1000x)
    
### 3. tensorflow 및 torch 버전 확인
만약 gpu 코드로 작성했지만 cpu로만 연산이 되거나, 아래 그림과 같이 device 종류가 XLA_CPU, XLA_GPU로 되어 있다면 tensorflow 버전이 잘못 설정되어있을 확률이 높다.

![](./GPU_인식_확인_방법/.figures/p11.png =1000x)

- tensorflow
gpu를 이용하기 위해서는 tensorflow가 아닌 tensorflow-gpu 라이브러리를 설치해야 하고 CUDA와 CuDNN에 맞는 tensorflow-gpu 버전은 https://www.tensorflow.org/install/source_windows#tested_build_configurations 에서 확인할 수 있습니다.

![](./GPU_인식_확인_방법/.figures/p12.png =1000x)

예를 들어 GPU 서버의 경우 CUDA 11.0과 cuDNN 8.0이 설치되어 있으므로 tensorflow-gpu-2.4.0이 필요합니다.
```
pip install tensorflow-gpu==2.4.0
```

- pytorch
pytorch(torch)는 홈페이지에서 OS, Package, CUDA 버전등을 선택하면 자동으로 pip 설치 명령어를 생성해줍니다. pytorch 홈페이지 주소는 https://pytorch.org/get-started/locally/ 입니다. 생성된 명령어를 jupyter notebook 또는 spark가 설치되어 있는 docker에 접속하여 입력하면 자동으로 설치됩니다.

![](./GPU_인식_확인_방법/.figures/p13.png =1000x)

![](./GPU_인식_확인_방법/.figures/p14.png =1000x)
        

## 3. cpu만 사용하는 코드에서 gpu 사용하는 코드로 변환 방법

### 1. tensorflow-gpu
tensorflow-gpu 라이브러리를 설치하였고 CUDA가 설치되었다면 기본적으로 GPU:0를 이용해 연산을 진행합니다.


만약 2개 이상의 GPU를 가지고 있고, 두개의 다른 method나 연산을 진행할 때는 다음과 같이 tf.device()를 이용해 어떤 gpu를 통해 연산할지 지정해줄 수 있습니다.

```
import tensorflow as tf

with tf.device("GPU:0"):
  get_something_from_dataset()
{} <= 여기에 들어가는 변수 또는 method는 GPU:0의 메모리에 저장됩니다.

with tf.device("GPU:1"):
  caculate_number_in_dataset()
{} <= 여기에 들어가는 변수 또는 method는 GPU:1의 메모리에 저장됩니다.
```
그렇기 때문에 GPU:0에 저장된 변수와 GPU:1에 저장된 변수는 서로 연산할 수 없고 만약 연산을 하기 위해서는 꼭 같은 device에 저장되어 있어야 합니다.

만약 하나의 학습모델을 2개 이상의 GPU를 이용해서 분산 훈련하고 싶다면 tf.distribute.MirroredStrategy()를 이용해야 합니다.

```
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  get_something_from_dataset()
  caculate_number_in_dataset()
```

tf.distribute.MirroredStrategy()을 공백으로 입력하면 tensorflow가 인식한 모든 GPU에 분산으로 변수와 method들을 복제하여 gpu 메모리에 저장합니다. 만약 특정 GPU에만 복사하고 싶다면 mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])와 같이 입력하시면 됩니다.

```
with mirrored_strategy.scope():
  get_something_from_dataset()

with mirrored_strategy.scope():
  caculate_number_in_dataset()
```
위와 같이 mirrored_strategy.scope()를 두개로 분리하여 적용할 경우 변수나 method가 꼬여서 계산되지 않는 경우가 생기므로 하나의 scope 구문내에서 모든 함수와 변수를 처리하길 권장드립니다.

### 2. pytorch
pytorch도 GPU가 인식되어있다면 GPU:0를 기본적으로 사용합니다.        

pytorch에서도 GPU를 이용하기 위해서는 tf.device와 똑같이 torch.device()를 이용합니다.

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

만약 pytorch에서 특정 GPU를 사용하고 싶다면 다음과 같이 이용하시면 됩니다.

```
device = torch.device('cuda:{특정 GPU Index}' if torch.cuda.is_available() else 'cpu')

예) device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
```

이후 gpu 메모리에 저장하고 싶은 변수 또는 모델 뒤에 다음 중 하나를 선택해서 선언해주시면 됩니다.

```
model1().cuda()
또는
model1().to(device)
또는
model1(device=device)
```

만약 하나의 학습모델을 2개 이상의 GPU를 이용해서 분산 훈련하고 싶다면 torch.nn.parallel을 이용해야 합니다.

```
model = nn.DataParallel(model, list({gpu Index들}))

GPU가 2개 설치된 GPU 서버의 경우를 예시로 들면
예) netG = nn.DataParallel(netG, list(range(2)))
또는 netG = nn.DataParallel(netG, list([0, 1]))
```

        
# Appendix

-tensorflow에서 gpu 2개 이상 사용하기 위한 방법
> https://www.tensorflow.org/tutorials/distribute/custom_training?hl=ko

