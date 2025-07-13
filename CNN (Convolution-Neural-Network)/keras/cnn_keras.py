import tensorflow as tf
import tensorflow_datasets as ds

# MNIST 데이터셋 로드 => cnn_tensorflow_v2.py와 동일
(train_ds, test_ds), ds_info = ds.load(
    'mnist', split=['train', 'test'], shuffle_files=True,
    as_supervised=True, with_info=True
)

#  데이터 전처리 => cnn_tensorflow_v2.py와 동일
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # [0, 1] 정규화
    image = tf.reshape(image, [28, 28, 1])  # image를 28 x 28 의 1채널 크기로 구성
    label = tf.one_hot(label, depth=10) # label을 원-핫 인코딩을 이용해 변환
    return image, label

# cnn_tensorflow_v2.py와 동일
batch_size = 100
train_ds = train_ds.map(preprocess).shuffle(10000).batch(batch_size)
test_ds = test_ds.map(preprocess).batch(batch_size)


'''
>> keras에서는 세 가지의 모델 생성 방법을 제공한다.
- sequential API: 레이어를 순서대로만 쌓는 간단한 방식의 모델 (프로토타입을 만들 때 주로 사용)
- functional API: 복잡한 네트워크 구조가 구성 가능한 모델
- subclassing API: tf.keras.Model을 상속받아 __init__()과 call()을 수동으로 정의하는 모델 (가장 자유로운 방식이며 연구용으로 주로 사용)
'''
class CNNModel(tf.keras.Model): # subclassing API를 사용하여 모델 정의
    def __init__(self):
        super(CNNModel, self).__init__()
        # 첫 번째 합성곱 층 구현 (필터 32개, 크기 3x3, padding 방식=same, relu 활성화 함수 사용)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        # 2x2 크기의 max pooling을 진행함으로써 출력 크기를 28x28에서 14x14로 축소
        self.pool1 = tf.keras.layers.MaxPooling2D(2, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, padding='same')    # 14x14 -> 7x7 다운샘플링

        self.flatten = tf.keras.layers.Flatten()    # 7x7x64인 3차원 특집맵을 1차원으로 펼침
        self.dense1 = tf.keras.layers.Dense(256, activation='relu') # 은닉층 구현 및 출력 노드 수를 256개로 설정
        self.dropout = tf.keras.layers.Dropout(0.3) # 30% 확률로 뉴런을 무작위로 제가 (== "keep_prob=0.7")

        self.dense2 = tf.keras.layers.Dense(10) # 출력층 10개

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)   # softmax 함수가 적용되지 않은 logits 출력
    
def main():
    model = CNNModel()
    # compile(): 케라스 라이브러리의 함수로써, 모델을 컴파일 하는 함수.
    #   ㄴ 모델을 학습시키기 전에 모델의 손실 함수, 최적화 방법, 평가 지표를 설정하는 역할
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),    # RMSProp + Momentum 개념이 결합된 Adam(Adaptive Moment Estimation) 알고리즘 사용
                  # Categorical Crossentropy는 softmax 함수 뒤에 cross-entropy loss를 붙인 형태이며,
                  # from_logits=True 는 loss 함수 내부에서 softmax를 자동적으로 적용해준다는 의미.
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=5)
    loss, acc = model.evaluate(test_ds)

    print(f"최종 정확도: {acc:.4f}")

if __name__ == "__main__":
    main()