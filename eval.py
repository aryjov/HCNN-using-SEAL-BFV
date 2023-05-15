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
x_test3 = np.array(x_test, np.longlong)
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
weights11 = weights1
weights22 = weights2
weights33 = weights3
weights1 = [[[[0 for k in range(5)] for j in range(1)] for h in range(5)] for t in range(5)]
weights2 = [[[[0 for k in range(50)] for j in range(5)] for h in range(5)] for t in range(5)]
weights3 = [[0 for k in range(10)] for j in range(800)]

for i in range(5):
    for l1 in range(5):
        for l2 in range(5):
            weights1[l1][l2][0][i] = batch_encoder.encode([int(weights11[l1][l2][0][i])] * slot_count)

for i in range(50):
    for fil in range(5):
        for l1 in range(5):
            for l2 in range(5):
                weights2[l1][l2][fil][i] = batch_encoder.encode([int(weights22[l1][l2][fil][i])] * slot_count)

for i in range(10):
    for j in range(800):
        weights3[j][i] = batch_encoder.encode([int(weights33[j][i])] * slot_count)

print("--- %s seconds --- Batch encoding weights" % (time.time() - start_time1)) # Не входит в общее время работы сети

start_time1 = time.time()

cinput1 = [[0 for k in range(28)] for j in range(28)]

for i in range(28):
    for j in range(28):
        cinput1[i][j] = [0 for l in range(16384)]
        for k in range(10000):
            cinput1[i][j][k] = x_test3[k][i][j]
        temp = batch_encoder.encode(cinput1[i][j])
        cinput1[i][j] = encryptor.encrypt(temp)

print("--- %s seconds --- Шифрование изображений" % (time.time() - start_time1))
        
lyr11 = [[[encryptor.encrypt(Plaintext("0")) for k in range(5)] for j in range(12)] for i in range(12)]

print(f'noise budget in lyr11[0][0][0]: {decryptor.invariant_noise_budget(lyr11[0][0][0])} bits')

start_time1 = time.time()

for i in range(5):
    for j in range(12):
        for k in range(12):
            for l1 in range(5):
                for l2 in range(5):
                    if weights11[l1][l2][0][i] != 0:
                        temp = evaluator.multiply_plain(cinput1[j * 2 + l1][k * 2 + l2], weights1[l1][l2][0][i])
                        lyr11[j][k][i] = evaluator.add(lyr11[j][k][i], temp)

print("--- %s seconds --- Свёртка 1" % (time.time() - start_time1))

start_time1 = time.time()

for i in range(5):
    for j in range(12):
        for k in range(12):
            lyr11[j][k][i] = evaluator.square(lyr11[j][k][i])
            evaluator.relinearize_inplace(lyr11[j][k][i], relin_keys)
            
print("--- %s seconds --- Активация 1" % (time.time() - start_time1))        

start_time1 = time.time()

lyr21 = [[[encryptor.encrypt(Plaintext("0")) for k in range(50)] for j in range(4)] for i in range(4)] 
for i in range(50):
    for j in range(4):
        for k in range(4):
            for fil in range(5):
                for l1 in range(5):
                    for l2 in range(5):
                        if weights22[l1][l2][fil][i] != 0:
                            temp = evaluator.multiply_plain(lyr11[j * 2 + l1][k * 2 + l2][fil], weights2[l1][l2][fil][i])
                            lyr21[j][k][i] = evaluator.add(lyr21[j][k][i], temp) 


print("--- %s seconds --- Свёртка 2" % (time.time() - start_time1))

start_time1 = time.time()

for i in range(50):
    for j in range(4):
        for k in range(4):
            lyr21[j][k][i] = evaluator.square(lyr21[j][k][i])
            evaluator.relinearize_inplace(lyr21[j][k][i], relin_keys)
       
print("--- %s seconds --- Активация 2" % (time.time() - start_time1))

lyr31 = []
for i in range(4):
    for j in range(4):
        for k in range(50):
            lyr31.append(lyr21[i][j][k])

lyr41 = [encryptor.encrypt(Plaintext("0")) for k in range(10)]

start_time1 = time.time()

for i in range(10):
    for j in range(800):
        if weights33[j][i] != 0:
            temp = evaluator.multiply_plain(lyr31[j], weights3[j][i])
            lyr41[i] = evaluator.add(lyr41[i], temp)

print("--- %s seconds --- Полносвязный" % (time.time() - start_time1))

noise = decryptor.invariant_noise_budget(lyr41[0])

for i in range(10):
    if decryptor.invariant_noise_budget(lyr41[i]) > noise:
        noise = decryptor.invariant_noise_budget(lyr41[i])
        
print(f'noise budget in lyr41: {noise} bits')        
print("--- %s seconds ---" % (time.time() - start_time))
           
    


output = np.zeros((10, 10000), dtype=np.double) 
for i in range(10):
    output[i] = np.array(batch_encoder.decode(decryptor.decrypt(lyr41[i]))[:10000], dtype = np.double)
output1 = np.argmax(output, axis=0) # в output1 сохранены предсказания для всего набора из 10000 изображений

print("--- %s seconds ---" % (time.time() - start_time))