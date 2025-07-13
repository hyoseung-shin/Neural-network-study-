import tensorflow as tf
import tensorflow_datasets as ds
import numpy as np

# MNIST 데이터셋 로드
(train_ds, test_ds), ds_info = ds.load(
    '''
    [tensorflow_datasets.load()의 매개변수 설명]
    - with_info=True : dataset 메타데이터가 포함된 ds.core.DatasetInfo 반환
    - as_supervised=True: 2-tuple 구조 (input, label) 구조로 반환
    '''
    'mnist', split=['train', 'test'], shuffle_files=True,
    as_supervised=True, with_info=True
)

#  데이터 전처리
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # image의 값들을 float32 형태로 변환(cast 함수의 역할) 후, 255.0으로 나눔으로써 [0, 1] 정규화
    image = tf.reshape(image, [28, 28, 1])  # image를 28 x 28 의 1채널 크기로 구성
    label = tf.one_hot(label, depth=10) # label을 원-핫 인코딩을 이용해 변환
    return image, label

batch_size = 100
'''
>> tf.data.Dataset pipline에서 학습 데이터셋을 전처리하고 섞고 나누는 작업 진행
    - train_ds.map(preprocess): train_ds의 각 요소에 preprocess 함수 적용
    - .shuffle(10000): 학습 데이터의 순서를 무작위로 섞음. 100000은 shuffle buffer size로, 이 크기만큼 데이터를 메모리에 올려두고 섞음.
    - .batch(batch_size): 지정한 크기만큼 데이터를 묶어 배치 단위로 만듦.
'''
train_ds = train_ds.map(preprocess).shuffle(10000).batch(batch_size)
test_ds = test_ds.map(preprocess).batch(batch_size)

def weight_initialize(shape):
    '''
    tf.Variable(
        initial_value=None, # 필수 매개변수 값으로써 초기값 설정
        trainable=True, # 학습 대상 파라미터 여부를 설정. True인 경우 optimizer가 이 변수의 값을 업데이트
        validate_shape=True,    # True일 경우, shape 변경 불가능
        caching_device=None,    # 변수값을 cache할 device
        name=None,  # 변수의 이름
        variable_def=None,  
        dtype=None, # 데이터의 타입 설정
        import_scope=None,
        constraint=None,
        shape=None, # 명시적 shape를 지정하며, 일반적으로 초기값에서 추론하여 설정함으로써 생략 가능
        synchronization=tf.VariableSynchronization.AUTO,    # 분산 학습에서 변수 동기화 설정
        aggregation=tf.VariableAggregation.NONE,    
        experimental_autocast=False
    )
    '''
    return tf.Variable(tf.random.normal(shape, stddev=0.01)) # stddev = standard deviation (표준편차)

class CNNModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W1 = weight_initialize([3, 3, 1, 32])  # = self.W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        self.W2 = weight_initialize([3, 3, 32, 64])
        self.W3 = weight_initialize([7 * 7 * 64, 256])
        self.W4 = weight_initialize([256, 10])

    def __call__(self, x, keep_prob):
        # Feature Extractor
        # 은닉층 01
        '''
        >> tf.nn.conv2d(): 이미지와 필터를 convolution하여 feature을 추출하는 계층을 구현하는 함수
        tf.nn.conv2d(
            input,  # 입력 텐서 (크기는 [batch, height, width, chaanels])
            filters,    # 필터 텐서
            strides,    # 슬라이딩 간격
            padding,    # 'SAME'은 입력과 출력 크기가 같도록 padding, 'VALID'는 패딩 없이 진행
            data_format='NHWC', # NHWC가 기본값, 또는 NCHW
            dilations=None, # 커널 간격으로써, 일반적으로 [1, 1, 1, 1]이며, dilated conv 사용 시 조절한다.
            name=None
        )

        '''
        x = tf.nn.conv2d(x, self.W1, strides=1, padding='SAME') # 3x3 필터, stride=1, 32채널 => 출력: 28 x 28 x 32
        '''
        >> tf.nn.relu(features, # 입력 텐서 (convolution 또는 dense 연산 결과)
                    name    # 선택)
        '''
        x = tf.nn.relu(x)
        '''
        >> tf.nn.max_pool2d(): 지역적으로 가장 큰 값을 뽑아서 공간 정보를 축소하는 연산
            ㄴ 특징 강조 및 계산량 감소가 주된 목적이다.
        tf.nn.max_pool2d(
            input,  # 입력 텐서 (크기는 [batch, height, width, channels])
            ksize,  # 풀링 영역의 크기를 의미 (형식은 [1, pool_height, pool_width, 1])
            strides,    # 풀링 슬라이딩 간격
            padding,    #  .conv2d()와 동일한 역할
            data_format='NHWC', 기본은 NHWC
            name=None
        )

        '''
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME') # 출력: 14 x 14 x 32

        # 은닉층 02
        x = tf.nn.conv2d(x, self.W2, strides=1, padding='SAME')  # 3x3 필터, stride=1, 64채널 => 출력: 14 x 14 x 64
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME') # 출력: 7 x 7 x 64

        # flatten
        x = tf.reshape(x, [-1, 7 * 7 * 64]) # 7 * 7 * 64 = 3,136개의 벡터
        # classifier
        # dense 층
        x = tf.matmul(x, self.W3)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, rate=1-keep_prob)

        x = tf.matmul(x, self.W4)   # output
        return x

def loss_function(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def accuracy_function(logits, labels):
    pred = tf.argmax(logits, axis=1)    # 모델의 출력(logits) 샘플 중 가장 높은 확률을 가진 값을 예측값으로 설정
    true = tf.argmax(labels, axis=1)    # 정답값
    return tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

def main():
    model = CNNModel()
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0  
        for batch_images, batch_labels in train_ds:
            '''
            >> tf.Gradient(): tensorflow의 자동 미분 기능을 사용하기 위한 일종의 "녹화 장치"
                - 코드 블록 내에서 실행되는 모든 텐서 연산을 기록해두고
                - 그 연산을 바탕으로 기울기를 자동으로 계산할 수 있도록 함.
            '''
            with tf.GradientTape() as tape: # tape는 아래 코드 내에서 발생하는 모든 연산을 기록하고, 나중에 해당 연산을 기반으로 gradient를 계산하는 데 도움.
                logits = model(batch_images, keep_prob=0.7) # 모델에 대한 예측값 저장
                loss = loss_function(logits, batch_labels)  # 예측값과 비교한 손실값 저장
            
            grads = tape.gradient(loss, model.trainable_variables)  # loss를 모델의 학습 가능한 파라미터들에 대해 미분하여 gradient 계산
            optimizer.apply_gradients(zip(grads, model.trainable_variables))    # gradient를 optimizer을 이용해 계산하여 zip() 함수를 이용하여 (gradient, 변수) 형태로 업데이트
            total_loss += loss.numpy()  # 에포크 전체 손실값을 집계하는데 사용
        print(f"Epoch {epochs + 1:02d}, Avg. Loss = {total_loss / len(train_ds):.3f}")

    print("최적화 완료!")

    total_acc = 0
    total_batches = 0

    for test_images, test_labels in test_ds:
        logits = model(test_images, keep_prob=1.0)
        acc = accuracy_function(logits, test_labels)
        total_acc += acc.numpy()
        total_batches += 1

    print(f"Accuracy {total_acc / total_batches:.4f}")


if __name__ == "__main__":
    main()