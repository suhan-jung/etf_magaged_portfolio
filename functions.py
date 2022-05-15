import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# tensorflow의 래핑 라이브러리인 keras에서 본 튜토리얼에 사용할 기능들
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Input

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# 일자별 종가인 dataframe을 받아서 nan값 제거, linear interpolation을 한후 로그일간수익율로 변환하여 반환한다.
def preprocessing(data):
    df = data.dropna(thresh=4)
    # 연휴에 따른 급격한 변화를 smoothing해주기 위해 interpolation
    df = df.interpolate(method='linear', limit_direction='forward')  
    df = df.dropna()
    dr = np.log(df).diff(1).dropna()
    return dr

# past/future array 분리 함수
def make_data_window(data, window_size_past=60, window_size_future=20):
    # sequential data와 과거데이터 수, 미래 데이터 수 를 받아서 과거데이터 ndarray, 미래데이터 ndarray를 반환한다.
    inputs_past = []
    inputs_future = []
    # print(len(data)-window_size_past-window_size_future)
    for i in range(len(data)-window_size_past-window_size_future):
        # print(i)
        inputs_past.append(data[i:i+window_size_past].copy())
        inputs_future.append(data[i+window_size_past:i+window_size_past+window_size_future].copy())
        
    np_inputs_past = np.array(inputs_past)
    np_inputs_future = np.array(inputs_future)
    return np_inputs_past, np_inputs_future

# over confidence를 제어할 조절 변수 정의
GAMMA_CONST = 0.1
# GAMMA_CONST = 0.001
REG_CONST = 0.1
# REG_CONST = 0.001

# custom loss function 정의
def markowitz_objective(y_true, y_pred):
    W = y_pred
    xf_rtn = y_true
    W = tf.expand_dims(W, axis=1)
    R = tf.expand_dims(tf.reduce_mean(xf_rtn, axis=1), axis=2)
    C = tfp.stats.covariance(xf_rtn, sample_axis=1)
    
    rtn = tf.matmul(W, R)
    vol = tf.matmul(W, tf.matmul(C, tf.transpose(W, perm=[0,2,1]))) * GAMMA_CONST
    reg = tf.reduce_sum(tf.square(W), axis=-1) * REG_CONST
    # print(f"rtn: {rtn}, vol: {vol}, reg: {reg}")
    objective = rtn - vol - reg
    
    return -tf.reduce_mean(objective, axis=0)

# 모델 생성 함수
def model_build_fit(xc_train, xf_train, xc_test, xf_test, epochs=100, batch_size=32):
    # LSTM으로 Markowitz 모델을 생성한다.
    scale_factor = 50.0
    xc_train_scaled = xc_train.astype('float32') * scale_factor
    xf_train_scaled = xf_train.astype('float32') * scale_factor
    xc_test_scaled = xc_test.astype('float32') * scale_factor
    xf_test_scaled = xf_test.astype('float32') * scale_factor

    
    # 입력 순서에 따른 상관성을 제거하기 위해 sklearn.utils의 함수를 이용해서 shuffle을 수행한다.
    xc_train_scaled, xf_train_scaled = shuffle(xc_train_scaled, xf_train_scaled)
    
    N_PAST = xc_train.shape[1]
    N_STOCKS = xc_train.shape[2]
    N_FUTURE = xf_train.shape[1]
    
    xc_input = Input(batch_shape=(None, N_PAST, N_STOCKS))
    h_lstm = LSTM(64, dropout=0.5, use_bias=True)(xc_input)
    y_output = Dense(N_STOCKS, activation='tanh')(h_lstm)

    # 특정 종목을 과도하게 매수하는 것을 방지하기 위해 위에서 tanh를 사용했다.(over confidence 방지용)
    # REG_CONST를 적용했기때문에 이미 고려된 사항이지만, 안전을 위해 추가했다.

    # 마코비츠의 최적 weights
    y_output = Activation('softmax')(y_output)

    model = Model(inputs=xc_input, outputs=y_output)
    # model.compile(loss=markowitz_objective, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
    model.compile(
        loss=markowitz_objective, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, decay=0.001)
        )
    # MPN을 학습하고 결과를 저장한다.
    history = model.fit(
        xc_train_scaled, 
        xf_train_scaled, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(xc_test_scaled, xf_test_scaled))
    # model.save(SAVE_MODEL)
    return model, history