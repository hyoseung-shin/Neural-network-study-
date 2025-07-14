import tensorflow as tf
import numpy as np
import time  # 시간 측정용 모듈

# 문자 사전 ()
char_arr = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # +1을 한 이유: range의 범위 특성 때문
'''
[num_dic 구성 원리]
>> enumerate(): 순서가 있는 자료형을 입력받았을 때, 인덱스 값을 포함하여 tuple 형태로 반환하는 함수
        ㄴ for문에 자주 사용됨
- enumerate(char_arr)은 char_arr의 인덱스와 값을 튜플 형태로 반환
- for i, n in enumerate(char_arr)에서 i는 인덱스, n은 문자를 의미.
- n: i for i, n in enumerate(char_arr)는 각 문자(n)을 키로 하고, 그 위치 인덱스(i)를 값으로 가지는 dictionary 생성
'''
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 시퀀스 데이터 
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool',
            'load', 'love', 'kiss', 'kind']

# 데이터 생성 -> 문자 형태인 데이터를 one-hot 인코딩된 3D 배열로 반환하는 함수
def make_batch(seq_data):
    input_batch = []    # 입력은 단어의 처음 세 글자
    target_batch = []   # 타켓은 단어의 마지막 글자

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]  # 단어의 마지막 글자를 제외한 글자를 꺼내 정수로 변환 ex) w, o, r -> [22, 14, 17]
        target = num_dic[seq[-1]]   # 단어의 마지막 글자를 꺼내 정수로 변환 ex) d -> [3]
        '''
        >> np.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')
        - (N x M) 크기의 2차원 행렬을 만들어 주는 함수 k에 따라 시작 위치가 달라짐.
        - M을 따로 설정하지 않는 경우, N x N의 정사각행렬이 구성됨. (따라서 N은 필수 요소)

        - np.eye(dic_len) -> 26 x 26 크기의 단위행렬 구성
        - np.eye(dic_len)[input] -> input에 들어있는 인덱스 위치의 행만 추출 -> one-hot encoding
            ex) input = [22, 14, 17] -> np.eye(26)[22], np.eye(26)[14], np.eye(26)[17] 각각 one-hot 벡터
                ㄴ (3, 26) 크기의 one-hot 인코딩된 시퀀스 제작
        '''
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    # input_batch와 target_batch를 numpy 배열로 변환
    #   ㄴ input_batch -> (10, 3, 26) 형태      /       target_batch -> (10, ) 형태
    return np.array(input_batch, dtype=np.float32), np.array(target_batch, dtype=np.int32)
    '''
    [list를 numpy 배열로 변환하는 이유]
    1. 텐서플로우는 입력값으로 tensor 값을 요구하는데, 이는 numpy 배열과 매우 유사한 텐서 구조를 사용하기 때문이다.
    2. 벡터 및 행렬 연산이 가능해진다. (list는 불가능 / numpy는 가능)
    3. 텐서플로우는 대부분 np.array로부터 텐서를 만들기 때문에 GPU 연산 또는 tensor로의 변환이 쉬움에 따라
        연산 효율이 증가한다.
    4. 구현된 데이터의 형상을 명확하게 제어할 수 있다. (.shpae의 속성을 이용)
    '''

# 하이퍼파라미터
n_input = dic_len
n_hidden = 128
n_step = 3  # word의 wor
n_class = dic_len
learning_rate = 0.01
epochs = 30

# 데이터
X_data, Y_data = make_batch(seq_data)
batch_size = X_data.shape[0]    # 3

# 변수 -> 내부의 게이트 연산을 위한 파라미터
'''
[n_hidden의 크기를 4배 증가하는 이유]
- LSTM 셀에는 input_gate(i), forget_gate(f), cell_candidate(g), output_gate (g)로
    4개의 게이트로 구성되기 때문이다.
- 따라서 각 입력인 x는 4개의 gate를 위한 가중치에 동시에 곱해져야하기 때문이다.
!!! 케라스와 파이토치에서는 위 과정을 LSTM 내부 처리를 자동화해주기 때문에 가중치 shape를
    직접 [n_input, 4 * n_hidden]으로 설정하는 과정은 필요 없다.!!!

[가중치를 random.normal()로 초기화하는 이유]
- 초기 가중치가 모두 동일할 경우, 모든 뉴런이 동일하게 학습되어버려 의미 없는 결과가 도출됨
    ㄴ 정규분포로 초기화하면 뉴런마다 서로 다른 초기값으로 시작되어 학습 가능
- 평균0, 분산 1은 너무 크지도 작지도 않은 값이라 기울기 폭발 또는 소실 문제를 감소시킴

[랜덤값을 Variable() 함수로 감싸는(?) 이유]
- tf.random.normal()은 단순히 숫자를 생성하는 용도.
- tf.Variable을 통해 학습 가능한 파라미터로 등록되어야, 역전파 시 gradient가 계산되고 업데이트도 가능하다.

[+ tensorflow에서 변수(variable)과 상수(constant)의 차이]
- 변수는 값의 변경이 가능(mutable value)한 반면, 상수는 값의 변경이 불가능(immutable value)하다.
- 따라서 변수는 값의 변경이 가능하고, 공유되고, 유지되는 특성 때문에 딥러닝 모델에을 훈련할 때,
    자동 미분 값의 역전파에서 가중치를 업데이트한 값을 저장하는데 사용된다.
    ㄴ 변수는 초기화를 필요로 한다.
'''
# LSTM 신경망 구현에 사용되는 가중치 및 편향값
Wx = tf.Variable(tf.random.normal([n_input, 4 * n_hidden])) # 평균과 표준편차를 설정하지 않을 경우, 기본값인 0과 1로 설정
Wh = tf.Variable(tf.random.normal([n_hidden, 4 * n_hidden]))
b = tf.Variable(tf.zeros([4 * n_hidden]))

# LSTM 신경망 출력값을 최종 출력으로 변환하기 위한 선형 계층(가중치 및 편향값)
Wo = tf.Variable(tf.random.normal([n_hidden, n_class]))
bo = tf.Variable(tf.zeros([n_class]))

# LSTM 수식 구현
'''
- keras와 pytorch에서는 LSTM 구현을 셀 내부적으로 저수준으로 구현하지 않는다.
- 아래 코드는 LSTM 셀 내부에서 시간 순서대로 데이터를 처리하는 로직으로,
tensorflow의 고수준 API 없이 LSTM을 수식 수준에서 구현한 것이다.

LSTM의 수식: 전통적인 LSTM 셀은 4개의 게잍와 2개의 상태를 가진다.
i_t : input gate    /   f_t : forget gate   /   o_t : output gate   /   g_t : cell candidate
c_t : 셀 상태       /   h_t : 히든 상태

gates = (x_t)*(W_x) + (h_(t-1))*(W_h)+b
[i_t, f_t, g_t, o_t] = split(gates)
i_t = sigmoid(i_t)    /   f_t = sigmoid(f_t)   /   o_t = sigmoid(o_t)   /   g_t = tanh(g_t)
c_t = (f_t)*(c_(t-1)) + (i_t)*(g_t)
h_t =  (o_t)*tanh(c_t)
'''
def lstm_forward(x):
    h = tf.zeros([batch_size, n_hidden])
    c = tf.zeros([batch_size, n_hidden])

    for t in range(n_step): # 시퀀스의 길이만큼 반복 (ex: 'wor'처럼 3글자가 입력이면 n_step=3)
        '''
        입력 텐서(x)에서현재 시간 스탭(t)에 해당하는 입력을 꺼낸다.
            ㄴ x의 shape이 [batch_size, 3, 26]이면 xt의 shape는 [batch_size, 26] (1개의 시점에 대한 입력 벡터)
        '''
        xt = x[:, t, :]
        '''
        입력값인 xt와 이전 히든 상태인 h를 각각 가중치인 Wx와 Wh에 연산을 진행한다.
        이를 통해 4개의 게이트 정보를 모두 한 번에 계산할 수 있다.
        gates의 shape: [batch_size, 4 * n_hidden]
        '''
        gates = tf.matmul(xt, Wx) + tf.matmul(h, Wh) + b
        '''
        gates는 하나의 큰 벡터임으로, 각 게이트에 할당하여 나누어 계산.
        매개변수 axis=1은 feature 차원(열)을 따라 각 게이트에 분할한다는 의미/
        '''
        i, f, g, o = tf.split(gates, num_or_size_splits=4, axis=1)
        '''
        i, f, o 세 게이트는 모두 시그모이드 활성화 함수를 사용함으로써 
        결과값이 0 ~ 1 범위로 나와서 얼마나 기억할지 또는 잊을지를 조절한다.
        '''
        i = tf.sigmoid(i)   # 얼마나 추가할지
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        '''
        g 게이트에는 tanh 함수를 적용함으로써 결과값이 -1 ~ +1 사이의 값으로
        도출됨에 따라 셀 상태로 추가할 정보를 생성한다.

        g_t는 이전 히든 상태를 통해 계산된 지금 이 순간 중요하다고 생각되는 정보이다.
        이를 input gate인 i_t로 조절해서 셀 상태에 추가할지 여부를 결정한다.
        '''
        g = tf.tanh(g)  # 무엇을 추가할지
        '''
        c_t인 cell state는 일종의 장기 기억으로, 시간이 흐르더라도 중요한 정보를 유지하도록 설계된 구조
        아래 수식은 새로운 셀 상태를 업데이트하는 수식이다.

        f * c: 기존의 셀 상태를 얼마나 유지할지 (forget)
        i * g: 새로운 정보를 얼마나 반영할지 (input)
        => 최종적으로 새로운 셀 상태 c가 계산된다.
        '''
        c = f * c + i * g
        '''
        hidden state인 h (출력)는 셀 상태의 tanh 값을 output gate로 조절한 것으로,
        최종 LSTM의 출력 벡터이다.
        '''
        h = o * tf.tanh(c)

    return h  # 마지막 h 반환

def main():
    # 옵티마이저 -> 가중치 업데이트
    # Adaptive Moment Estimation 최적화 알고리즘을 사용하여 학습률을 기반으로
    # 모델의 가중치를 업데이트할 수 있는 옵티마이저 객체를 만듦.
    optimizer = tf.optimizers.Adam(learning_rate)

    # === 학습 시간 측정 시작 ===
    start_train = time.time()

    # 학습 루프
    for epoch in range(epochs):
        '''
        >> with tf.GradientTape() as tape:
        - tensorflow의 자동 미분 명령어? 코드?
        - 해당 코드 블록 내에서 실행되는 연산은 모든 변수에 대한 기울기를 자동으로 추적
        - 이후에 tape.gradient(loss, variables)를 호출해서 역전파에 필요한 미분값 계산 가능

        >> with 형식
        - 파일이나 리소스 관리와 관련된 작업을 간편하게 수행할 수 있도록 하는 구문
        - 리소스 해제 자동화 & 가독성과 간결성 & 예외 처리 간소화 & 코드 일관성에 유리

        with expression as variable
        '''
        with tf.GradientTape() as tape:
            # 입력값 X_data를 사용자가 정의한 lstm_forward 셀을 통해 입력을 처리 => 순전파 처리 단계
            # 반환되는 h 값은 마지막 hidden state
            h = lstm_forward(X_data)
            '''
            * logit이란 어떤 사건의 성공 확률을 나타내는 P(A)와 실패 확률을 나타내는 1-P(A)의 비율의 로그값

            >> 출력층 (선형 계층) 연산을 진행하는 코드 (softmax 함수 적용 이전의 logit)
                ㄴ LSTM이 뽑아낸 feature vector를 통해 최종적으로 26개의 클래스(알파벳) 중 하나를 예측.
            - h (LSTM의 출력값) -> shape: (batch_size, hidden_dim)
            - Wo (출력 가중치의 행렬 -> shape: (hidden_dim, n_class)
            - bo (bias 벡터) -> shape: (n_class, )
            - logits의 shape: (batch_size, n_class)
            '''
            logits = tf.matmul(h, Wo) + bo
            '''
            >> logits(예측값)과 Y_data(정답 레이블)를 비교해서 손실을 계산하는 함수
            - sparse: 정답이 정수형인 인덱스일 때 사용
                ㄴ 입력 시퀀스에서 target은 알파벳을 인덱스로 바꾼 정수형 라벨이다.
                ㄴ 이럴 때 정답을 일일이 one-hot 인코딩으로 변환하면 메모리 낭비가 심해짐.
            - 내부적으로 softmax 계산(벡터값을 확률분포로 변환)과 
              cross-entropy 계산(예측값과 정답값 간의 차이를 측정)을 동시에 진행
            - 반환 형태는 각 샘플의 손실 값인 (batch_size, )
            
            >> reduce_mean(): 모든 배치에 대한 평균 손실을 계산하여 스칼라값 1개만을 반환
            '''
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=Y_data, logits=logits))

        '''
        loss에 대한 각 파라미터의 미분값을 계산 후, optimizer에 반영하여
        업데이트함으로써 학습을 계속하여 진행.

        >> grads = tape.gradient(loss, [Wx, Wh, b, Wo, bo])
        - loss 값을 각 파라미터 (Wx, Wh, b, Wo, bo)에 대해 미분해서 각 변수의 기울기를 계산
        => 현재 손실이 가중치 각각에 얼마나 민감하게 반응하는지를 수치로 구하는 것

        >> optimizer.apply_gradients(zip(grads, [Wx, Wh, b, Wo, bo]))
        - 계산된 grads를 각 변수에 적용하여 업데이트
        '''
        grads = tape.gradient(loss, [Wx, Wh, b, Wo, bo])
        optimizer.apply_gradients(zip(grads, [Wx, Wh, b, Wo, bo]))

        print(f"Epoch {epoch+1:02d}, Loss: {loss.numpy():.6f}")

    end_train = time.time()
    train_duration = end_train - start_train
    print(f"\n총 학습 시간: {train_duration:.4f}초")

    # === 예측 시간 측정 시작 ===
    start_predict = time.time()

    h = lstm_forward(X_data)
    logits = tf.matmul(h, Wo) + bo
    '''
    각 입력 샘플에 대해 예측된 마지막 문자 인덱스를 얻는다.
        ㄴ logits의 shape: (batch_size, num_classes)
    argmax()는 각 샘플마다 가장 점수가 높은 클래스(알파벳 인덱스)를 반환
    '''
    pred = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
    # 모델의 예측이 정답과 얼마나 일치하는지를 확인 -> 맞으면 True, 틀리면 False를 반환 후 맞춘 개수의 비율을 계산
    accuracy = np.mean(pred == Y_data)

    end_predict = time.time()
    predict_duration = end_predict - start_predict

    predict_words = [seq[:3] + char_arr[p] for seq, p in zip(seq_data, pred)]

    # 출력
    print("\n=== 예측 결과 ===")
    print("입력값 :", [w[:3] + ' ' for w in seq_data])
    print("예측값 :", predict_words)
    print("정확도 :", accuracy)
    print(f"예측 시간: {predict_duration:.6f}초")

if __name__ == "__main__":
    main()