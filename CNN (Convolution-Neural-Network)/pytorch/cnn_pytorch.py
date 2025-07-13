import torch    # Tensor 등의 다양한 수학 함수가 포함된 라이브러리
import torch.nn as nn   # 신경망을 만드는 데 필요한 구조나 레이어(CNN, ReLU, Loss 등) 정의에 사용되는 라이브러리
import torch.nn.functional as F
import torch.optim as optim # Adam, SGD 등의 파라미터 최적화 알고리즘 구현을 위한 라이브러리
from torch.utils.data import DataLoader # Gradient Descent 계열의 반복 연산을 할 때 사용하는 미니배치용 유틸리티 함수가 포함된 라이브러리

'''
>> torchvision: pytorch에서 이미지 및 비디오 데이터를 다루기 위한 라이브러리
    - transform: 다양한 이미지 전처리 작업을 수행하는 라이브러리
        ㄴ ex) RandomCrop(), RandomRotation(), Resize()
    - dataset: torch에서 제공하는 데이터셋들이 모여있는 라이브러리
        ㄴ ex) COCO, MNIST, Fashion-MNIST, ImageNet, ...
'''
from torchvision import datasets, transforms


batch_size = 100
'''
- transforms.Compose(): 다양한 Data augmentation을 한꺼번에 손쉽게 해주는 클래스
- transforms.ToTensor(): 이미지(PIL 이미지 또는 numpy 배열)를 pytorch tensor 구조(channel x height x width)로 변환시켜주는 클래스
'''
transform = transforms.Compose([transforms.ToTensor(), ])   # 정규화 진행

train_ds = datasets.MNIST(root='./mnist/data', train=True, download=True, transform=transform)  # 매개변수 transform은 이미지를 tensor에 맞게 조정하기 위해 생성
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)    # DataLoader은 배치 기반의 딥러닝 모델 학습을 위해서 미니 배치를 만들어주는 역할

test_ds = datasets.MNIST(root='./mnist/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    # python에서 self의 역할
    def __init__(self):
        super(CNNModel, self).__init__()
        # nn.Conv2d(in_channels(input 텐서의 차원과 관련된 변수(int)), 
        #           out_channal(output 텐서의 채널 차원 >> convolution 계층의 커널 개수 (int)),
        #           kernel_size(kernel 단면의 크기 (int ot tuple)),
        #           stride_size(stride 값 / int 값이라면 가로, 세로 동일한 간격으로 연산 진행, tuple 값이라면 지정한 가로, 세로 간격으로 연산 진행))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 28 x 28 -> 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 14 x 14 -> 14 x 14

        # nn.Linear(): 선형 변환 (Linear Transformation)을 진행
        self.fc1 = nn.Linear(7 * 7 * 64, 256)   # fc는 Fully Connected Layers(완전 연결 신경망)의 약자
        self.fc2 = nn.Linear(256, 10)
        ## nn.Dropout이 받은 인자 값은 p값으로, 뉴런을 신경망에서 "제외"시키는 확률을 의미
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, keep_prob=1.0):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # max_pool1d()와 max_pool2d()의 차이 작성하기
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        '''
        >> view(): 특정 배열을 원하는 차원의 배열로 변경해주는 역할 (reshape 함수와 역할이 동일)
        >> view() vs reshpae()
            - view(): contiguous가 아닌 일부 tensor에 대해 사용이 제한될 수 있다.
            - reshape(): contiguous 여부에 상관없이 사용할 수 있다는 특징 (view보다 범용성이 더욱 높음
            
            * contiguous: 자료 내 "data"가 연속적인 것을 의미.
                ㄴ ex) 4, 6, 2, 7 -> not contiguous     /   2, 4, 6, 8 -> contiguous
        '''
        x = x.view(-1, 7 * 7 * 64)  # -1은 들어가야 할 남은 차원이 자동으로 할당

        x = F.relu(self.fc1(x))

        '''
        >> torch.nn.functional.dropout(input, p, training)
            - input: 적용할 신경망
            - p: 드롭할 확률(drop probability) > keep_prob는 유지할 확률이기에 해당 코드에서는 "1 - keep_prob"으로 적용
            - training: 학습할 때만 사용할 것인지에 대한 bool 매개변수
                ㄴ training=self.training은 학습 시에는 dropout이 활성화되고, 평가 시에는 비활성화 되도록 설정
        '''
        x = F.dropout(x, p=1-keep_prob, training=self.training) # pytorch에서는 "1-keep_prob"으로 기입하는 이유는?
        x = self.fc2(x)
        
        return x
    
def accuracy_function(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).float()
    return correct.sum() / len(correct)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    >> .to() 메소드의 의미: PyTorch 텐서나 모델을 다른 장치 또는 자료형으로 변환하는 함수
        - 기본 문법: tensor.to(device=None, dtype=None, non_blocking=False, copy=False)
            ㄴ device: 텐서를 이동할 장치를 지정
            ㄴ dtype: 텐서의 자료형을 변경
            ㄴ non_blocking: 데이터 복사를 비동기(asynchronous)로 수행할 수 있도록 시도
            ㄴ copy: 텐서가 이미 지정한 device와 dtype에 있다 하더라도, 복사본을 새로 만들지 여부 결정
        - 간단한 문법: tensor.to(device)
    '''
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            '''
            >> images와 labels 데이터에도 to(device)를 사용하는 이유
                - PyTorch에서는 Tensor와 모델(nn.Module)이 같은 장치에 있어야 연산이 가능
                    ㄴ 그렇지 않으면 연산 중 RuntimeError 발생
                - 모델이 to(device)의 위치에 있기에 images Tensor와 labels Tensor를 device로 이동
            '''
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()   # 기존 gradient 초기화
            outputs = model(images, keep_prob=0.7)
            loss = criterion(outputs, labels)
            loss.backward() # 역전파(gradient 계산)를 수행하는 명령어 (손실 함수를 기준으로 모든 학습 가능한 파라미터들에 대해 미분을 계산하는 함수)
            optimizer.step()    # PyTorch에서 gradient를 사용하여 모델의 가중치를 실제로 업데이트하는 함수

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1:02d}, Avg.Loss = {avg_loss:.3f}")

    print("최적화 완료!")

    model.eval()    # PyTorch에서 모델을 평가 모드로 전환하는 함수 > 학습과 평가 단계에서 동작 방식이 달라지는 레이어들을 올바르게 반영 (ex. dropout)
    total_acc = 0
    total_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, keep_prob=1.0)
            acc = accuracy_function(outputs, labels)
            total_acc += acc.item()
            total_batches += 1
    
    print(f"Accuracy {total_acc / total_batches:.4f}")

if __name__ == "__main__":
    main()