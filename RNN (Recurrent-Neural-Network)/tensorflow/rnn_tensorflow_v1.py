import tensorflow as tf
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load',
            'love', 'kiss', 'kind']

def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch

learning_rate = 0.01
n_hidden = 128
total_epoch = 30
n_step = 3
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=model, labels=Y))
optimizer - tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    '''
    [언더스코어 (underscore, _)는 아래와 같을 때 사용된다.]
    - 인터프리터(Interpreter)에서 마지막 값을 저장할 때
    - 값을 무시하고 싶을 때 (흔히 “I don’t care"라고 부른다.)
    - 변수나 함수명에 특별한 의미 또는 기능을 부여하고자 할 때
    - 국제화(Internationalization, i18n)/지역화(Localization, l10n) 함수로써 사용할 때
    - 숫자 리터럴값의 자릿수 구분을 위한 구분자로써 사용할 때
    '''
    _, loss = sess.run([optimizer, cost], 
                       feed_dict={X: input_batch, Y:target_batch})
    print('Epoch: ', '%04d' % (epoch + 1), 'cost', '{:.6f}'.format(loss))
print('최적화 완료!')

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)
predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X:input_batch, Y:target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값: ', [w[:3] + ' ' for w in seq_data])
print('예측값: ', predict_words)
print('정확도: ', accuracy_val)
