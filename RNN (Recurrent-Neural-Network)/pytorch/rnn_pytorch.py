import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# 문자 사전
char_arr = [chr(i) for i in range(ord('a'), ord('z') + 1)]
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 시퀀스 데이터
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool',
            'load', 'love', 'kiss', 'kind']

# 데이터 생성 함수
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return torch.tensor(input_batch, dtype=torch.float32), torch.tensor(target_batch, dtype=torch.long)

# 하이퍼파라미터
n_input = dic_len
n_hidden = 128
n_step = 3
n_class = dic_len
learning_rate = 0.01
epochs = 30

# 데이터 준비
X_data, Y_data = make_batch(seq_data)
# numpy에서는 .shape(), pytorch에서는 .size()
batch_size = X_data.size(0)

# 모델 정의
class LSTMModel(nn.Module):
    '''
    [tensorflow가 아닌 pytorch에서만 생성자를 사용하는 이유]
    1. PyTorch는 객체 지향 프로그래밍 스타일로 모델을 정의하기 때문이다.
        ㄴ PyTorch는 "낮은 수준의 제어와 유연한 구조"를 핵심 철학을 가지고 있기 때문에
            모델의 구조를 클래스로 정의하고, 학습 과정/forwar 연산/조건부 로직 등을 개발자가 직접 제어할 수 있게 한다.
        ㄴ 핵심 철학은 유연한 모델의 정의 / 사용자 맞춤형 동작 (forward() 커스터마이징) / 객체 지향은 "모델 = 객체"라는 자연스러운 추상화
    2. Keras에서는 내부적으로 모델 구조를 자동으로 처리해주는 고수준 API를 제공하기 때문에
        별도의 클래스를 만들 필요 없이 Sequential 또는 Functional API로 모델 구성이 가능하기 때문이다.
        ㄴ __init__() 생성자를 tensorflow에서도 사용할 수는 있지만 선택적.
    '''
    def __init__(self):
        super(LSTMModel, self).__init__()   # 부모 클래스 nn.Module 초기화
        '''
        >> nn.LSTM(): LSTM 층 정의
        nn.LSTM(
            input_size,     # 입력 특징의 차원 (one-hot 벡터 크기 등)
            hidden_size,    # LSTM 은닉 상태의 차원 수
            num_layers=1,   # LSTM 레이어의 개수 (기본값: 1)
            bias=True,      # bias 사용 여부 (기본값: True)
            batch_first=False,  # 입력 shape를 (batch, seq, input)로 처리할지 여부 (True 권장)
                                    ㄴ False일 경우 (seq_len, batch, input_size)로 처리
            dropout=0.0,    # 훈련 시 레이어 사이에 적용할 dropout 비율
            bidirectional=False # 양방향 LSTM인지에 대한 여부 (True일 경우 forward + backward)
        )
        '''
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden, batch_first=True)
        # LSTM의 출력인 hidden state를 class 수 만큼으로 분류
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, x):   # 입력값인 x를 받아 모델의 logits(예측)을 반환
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 시점의 출력
        out = self.fc(out)
        return out  # softmax 함수 적용 없이 class별 분류 점수 반환

def main():
    model = LSTMModel()
    '''
    >> optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        - model.parameters(): 학습할 파라미터들 (모델의 weight, bias 등)
            ㄴ pytorch에서 nn.Module을 상속한 모델 내부의 모든 학습 가능한 파라미터들을 반환
        - lr=learning_rate: 학습률
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    '''
    [CrossEntropyLoss()의 특징]
    - 내부적으로 log(softmax(logits)) 계산
    - y_true는 정수 인덱스, y_pred는 logits 형태
    '''
    loss_fn = nn.CrossEntropyLoss()

    # === 학습 시간 측정 시작 ===
    start_train = time.time()

    # 학습 루프
    for epoch in range(epochs):
        model.train()   # 학습 모드
        '''
        >> optimizer.zero_grad() 
        - 이전 단계에서 누적된 기울기를 초기화
        - why? .backward() 호출 시 기울기를 누적한다.
               따라서 새로운 기울기 계산을 하기 전에는 반드시 .zero_gard()로 이전 값을 초기화 진행
        - 만약 zero_gard()를 생략하면 이전 배치에서 계산된 gradient가 다음 배치에 덧셈되어 계산 누적
            ㄴ 이는 의도하지않은 기울기 폭발 또는 학습 이상의 원인이 됨
        '''
        optimizer.zero_grad()   
        logits = model(X_data)  # model의 forward 호출
        loss = loss_fn(logits, Y_data)  # 정답과 예측값을 비교하여 loss 계산
        loss.backward() # 현재 배치에 대한 기울기 계산
        optimizer.step()    # optimizer에 파라미터 업데이트 진행

        print(f"Epoch {epoch+1:02d}, Loss: {loss.item():.6f}")

    end_train = time.time()
    train_duration = end_train - start_train
    print(f"\n총 학습 시간: {train_duration:.4f}초")

    # === 예측 시간 측정 시작 ===
    start_predict = time.time()

    model.eval()    # 모델을 평가 모드로 전환 (dropout 해제, BatchNorm 등)
                    #   ㄴ BatchNorm: 훈련 중에는 배치 통계, 평가 시에는 저장된 평균-분산 사용
    with torch.no_grad():   # 자동 미분 기능 비활성화 -> 평가할 때는 필요 없기 때문에 메모리 절약 + 속도 향상 취지
        logits = model(X_data)
        # argmax(): 가장 큰 값을 가지는 인덱스(클래스)를 반환
        #   ㄴ dim=1은 클래스 차원을 의미 (즉, 알파벳 26개 중에서 가장 높은 점수의 인덱스를 선택)
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == Y_data).float().mean().item()

    end_predict = time.time()
    predict_duration = end_predict - start_predict

    predict_words = [seq[:3] + char_arr[p] for seq, p in zip(seq_data, pred.numpy())]

    # 출력
    print("\n=== 예측 결과 ===")
    print("입력값 :", [w[:3] + ' ' for w in seq_data])
    print("예측값 :", predict_words)
    print("정확도 :", accuracy)
    print(f"예측 시간: {predict_duration:.6f}초")


if __name__ == "__main__":
    main()