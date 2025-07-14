import tensorflow as tf
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

    return np.array(input_batch, dtype=np.float32), np.array(target_batch, dtype=np.int32)

def main():
    # 하이퍼파라미터
    n_input = dic_len
    n_hidden = 128
    n_step = 3
    n_class = dic_len
    learning_rate = 0.01
    epochs = 30

    # 데이터 준비
    X_data, Y_data = make_batch(seq_data)   # X_data는 (10, 3, 26) / Y_data는 (10, )
    '''
    >> shape
    - numpy에서 주어진 행렬의 열과 행의 개수를 알려주는 함수
        ㄴ ex) 2 x 3 행렬의 경우, shape의 출력값은 (2, 3)
    '''
    batch_size = X_data.shape[0]    # 10

    # 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_step, n_input)), # 입력 데이터의 shape 지정
        tf.keras.layers.LSTM(n_hidden), # 순환 신경망 계층; 128 차원의 hidden state 출력
        tf.keras.layers.Dense(n_class)  # 출력층; 알파벳 26개 중 하나를 예측
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    '''
    SparseCategoricalCrossentropy() 매개변수에서 from_logits=True로 설정했기 때문에
    출력에 softmax 함수를 직접 적용하지 않은 logits 형태로 처리

    [tf.nn.sparse_softmax_cross_entropy_with_logits (1) vs tf.keras.losses.SparseCategoricalCrossentropy (2)]
    - 입력 형식: (1)은 logits(softmax 적용 X), labels(정수 인덱스)이고,
                (2)는 y_true와 y_pred (softmax 적용 여부 선택 가능)이다.
    - softmax 적용: (1)은 내부적으로 자동 softmax 적용 (입력은 logits)이며,
                    (2)는 softmax 적용 여부를 from_logits=True/False로 명시한다.
    - 반환값: (1)은 각 샘플별 loss (벡터값)을 반환하며,
              (2)는 평균된 loss 또는 샘플별 loss를 반환한다.
    - 사용 방식: (1)은 한 번만 계산용으로 직접 호출하며,
                 (2)는 클래스로 생성 후, .call() 또는 함수처럼 사용한다.

    => 기능적으로는 동일하지만, (1)의 경우 함수 형태의 단발성 호출이며,
        (2)는 객체 생성 후 반복 사용 가능함으로써 keras 모델 학습 루프에서 주로 사용된다.
    
    >> tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,  # True일 경우, 입력이 softmax 적용 이전의 logits 이용
        reduction='auto',   # 평균을 낼지, 합을 낼지를 결정 ('auto', 'sum', 'none' 등)
        name='sparse_categorical_crossentropy'  # 손실 함수의 이름 (그래프 시각화 용도))
    - one-hot 대신 정수 인덱스를 사용 -> 따라서 별도의 one-hot encoding 과정이 필요 없음.
        ㄴ one-hot 벡터를 사용하는 손실 함수로는 CategoricalCrossentropy가 있다.
    - 분류 문제에 적합한 함수이며, 정답 포멧은 정수 인덱스, 예측 포멧은 softmax 함수 또는 logits이다.

    '''
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # === 학습 시간 측정 시작 ===
    start_train = time.time()

    # 학습 루프
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # 모델에 입력값인 X_data와 출력 결과인 logits를 얻는다.
            logits = model(X_data, training=True)
            # 정답과 예측값을 비교하여 loss를 계산
            loss = loss_fn(Y_data, logits)
        
        '''
        [[Wx, Wh, b, Wo, bo] (1) vs model.trainable_variables (2)]
        - 구현 스타일: (1)은 저수준 코드임에 따라 수동 구현이며,
                       (2)는 고수준 API임에 따라 자동 구현된다.
        - 사용 편의성: (1)은 어려지만, 유연성이 있고,
                       (2)는 간편하고 직관적이지만, (1)에 비해 유연성이 떨어진다.
        - 확장성: (1)은 모든 계층을 직접 구현해야함에 따라 확장성이 낮지만 (재사용 어려움),
                  (2)는 다양한 레이어와의 조합이 가능함에 따라 확장성이 높다 (재사용 good).
        '''
        grads = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Epoch {epoch+1:02d}, Loss: {loss.numpy():.6f}")

    end_train = time.time()
    train_duration = end_train - start_train
    print(f"\n총 학습 시간: {train_duration:.4f}초")

    # === 예측 시간 측정 시작 ===
    start_predict = time.time()

    logits = model(X_data, training=False)
    pred = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
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