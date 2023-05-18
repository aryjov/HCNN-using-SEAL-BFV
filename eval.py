from seal import *
from seal_helper import *
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = tf.math.round((x_test /255) * 4)
x_test_ = np.array(x_test, np.int8)
y_test_cat = to_categorical(y_test, 10)


#------------------------------------------------------------------------------------------------
model = keras.models.load_model('HCNN') # загрузка сети, обученной в train.py
#------------------------------------------------------------------------------------------------

weights1 = np.array(model.get_weights()[0], np.int8)
weights2 = np.array(model.get_weights()[1], np.int8)
weights3 = np.array(model.get_weights()[2], np.int8)

#----------------------------------------------------------------------------------------------------------------



import time
start_time = time.time()

start_time1 = time.time()

parms = EncryptionParameters(scheme_type.bfv)
parms.set_poly_modulus_degree(16384)
parms.set_coeff_modulus(CoeffModulus.Create(
    16384, [30, 40, 40, 40, 40, 50]))            
parms.set_plain_modulus(PlainModulus.Batching(16384, 39))

context = SEALContext(parms)
print_parameters(context)
keygen = KeyGenerator(context)
secret_key = keygen.secret_key()
public_key = keygen.create_public_key()
relin_keys = keygen.create_relin_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

batch_encoder = BatchEncoder(context)
slot_count = batch_encoder.slot_count()

print("--- %s seconds --- Задание контекста, параметров, генерация ключей,..." % (time.time() - start_time1))

start_time1 = time.time()
weights1_ = weights1
weights2_ = weights2
weights3_ = weights3
weights1 = [[[[0 for k in range(5)] for j in range(1)] for h in range(5)] for t in range(5)]
weights2 = [[[[0 for k in range(50)] for j in range(5)] for h in range(5)] for t in range(5)]
weights3 = [[0 for k in range(10)] for j in range(800)]

for i in range(5):
    for l1 in range(5):
        for l2 in range(5):
            weights1[l1][l2][0][i] = batch_encoder.encode([int(weights1_[l1][l2][0][i])] * slot_count)

for i in range(50):
    for fil in range(5):
        for l1 in range(5):
            for l2 in range(5):
                weights2[l1][l2][fil][i] = batch_encoder.encode([int(weights2_[l1][l2][fil][i])] * slot_count)

for i in range(10):
    for j in range(800):
        weights3[j][i] = batch_encoder.encode([int(weights3_[j][i])] * slot_count)

print("--- %s seconds --- Batch encoding weights" % (time.time() - start_time1)) # Не входит в общее время работы сети

start_time1 = time.time()

cinput = [[0 for k in range(28)] for j in range(28)]

for i in range(28):
    for j in range(28):
        cinput[i][j] = [0 for l in range(16384)]
        for k in range(10000):
            cinput[i][j][k] = x_test_[k][i][j]
        temp = batch_encoder.encode(cinput[i][j])
        cinput[i][j] = encryptor.encrypt(temp)

print("--- %s seconds --- Шифрование изображений" % (time.time() - start_time1))
        
lyr1 = [[[encryptor.encrypt(Plaintext("0")) for k in range(5)] for j in range(12)] for i in range(12)]

print(f'noise budget fresh ciphertext: {decryptor.invariant_noise_budget(lyr1[0][0][0])} bits')

start_time1 = time.time()

for i in range(5):
    for j in range(12):
        for k in range(12):
            for l1 in range(5):
                for l2 in range(5):
                    if weights1_[l1][l2][0][i] != 0:
                        temp = evaluator.multiply_plain(cinput[j * 2 + l1][k * 2 + l2], weights1[l1][l2][0][i])
                        lyr1[j][k][i] = evaluator.add(lyr1[j][k][i], temp)

print("--- %s seconds --- Свёртка 1" % (time.time() - start_time1))

start_time1 = time.time()

for i in range(5):
    for j in range(12):
        for k in range(12):
            lyr1[j][k][i] = evaluator.square(lyr1[j][k][i])
            evaluator.relinearize_inplace(lyr1[j][k][i], relin_keys)
            
print("--- %s seconds --- Активация 1" % (time.time() - start_time1))        

start_time1 = time.time()

lyr2 = [[[encryptor.encrypt(Plaintext("0")) for k in range(50)] for j in range(4)] for i in range(4)] 
for i in range(50):
    for j in range(4):
        for k in range(4):
            for fil in range(5):
                for l1 in range(5):
                    for l2 in range(5):
                        if weights2_[l1][l2][fil][i] != 0:
                            temp = evaluator.multiply_plain(lyr1[j * 2 + l1][k * 2 + l2][fil], weights2[l1][l2][fil][i])
                            lyr2[j][k][i] = evaluator.add(lyr2[j][k][i], temp) 


print("--- %s seconds --- Свёртка 2" % (time.time() - start_time1))

start_time1 = time.time()

for i in range(50):
    for j in range(4):
        for k in range(4):
            lyr2[j][k][i] = evaluator.square(lyr2[j][k][i])
            evaluator.relinearize_inplace(lyr2[j][k][i], relin_keys)
       
print("--- %s seconds --- Активация 2" % (time.time() - start_time1))

lyr3 = []
for i in range(4):
    for j in range(4):
        for k in range(50):
            lyr3.append(lyr2[i][j][k])

lyr4 = [encryptor.encrypt(Plaintext("0")) for k in range(10)]

start_time1 = time.time()

for i in range(10):
    for j in range(800):
        if weights3_[j][i] != 0:
            temp = evaluator.multiply_plain(lyr3[j], weights3[j][i])
            lyr4[i] = evaluator.add(lyr4[i], temp)

print("--- %s seconds --- Полносвязный" % (time.time() - start_time1))

noise = decryptor.invariant_noise_budget(lyr4[0])

for i in range(10):
    if decryptor.invariant_noise_budget(lyr4[i]) < noise:
        noise = decryptor.invariant_noise_budget(lyr4[i])
        
print(f'noise budget in output ciphertext: {noise} bits')        
print("--- %s seconds ---" % (time.time() - start_time))
           
    


output = np.zeros((10, 10000), dtype=np.double) 
for i in range(10):
    output[i] = np.array(batch_encoder.decode(decryptor.decrypt(lyr4[i]))[:10000], dtype = np.double)
output1 = np.argmax(output, axis=0) # в output1 сохранены предсказания для всего набора из 10000 изображений

print("--- %s seconds ---" % (time.time() - start_time))
