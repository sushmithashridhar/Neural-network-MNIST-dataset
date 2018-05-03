import numpy as np
import gzip
from sklearn.metrics import confusion_matrix

Learning_rate = 0.1
Momentum = 0.9

f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
train_images = np.frombuffer(f.read(), np.uint8, offset=16)
train_images = train_images.reshape(-1, 1, 28, 28)
train_images = train_images.reshape(train_images.shape[0],784)
train_images = train_images / np.float32(255)
train_images = np.append(train_images,np.ones((60000,1)),axis = 1)

f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
train_labels = train_labels.reshape(60000,1)

train_labels_final = np.full((60000, 10), 0.1, dtype=float)

for row in range(len(train_labels)):
    index = train_labels[row].astype(int)
    train_labels_final[row][index[0]] += 0.8

weights2 = np.random.uniform(-0.05, 0.05, size=(785,20))
deltaO = np.zeros((21,10))
deltaH = np.zeros((20,785))
weights3 = np.random.uniform(-0.05, 0.05, size=(21,10))
accuracy = 0
h6 = np.zeros((60000,1))
finding_accuracy_train = np.zeros((60000,1))
finding_accuracy_test = np.zeros((10000,1))
h3_with_bias = np.zeros(21)
h3_with_bias[0] = 1


f = gzip.open("t10k-images-idx3-ubyte.gz", 'rb')
test_images = np.frombuffer(f.read(), np.uint8, offset=16)
test_images = test_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(test_images.shape[0],784)
test_images = np.append(test_images,np.ones((10000,1)),axis = 1)


f = gzip.open("t10k-labels-idx1-ubyte.gz", 'rb')
test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
test_labels = test_labels.reshape(10000,1)

test_labels_final = np.full((10000, 10), 0.1, dtype=float)

for row in range(len(test_labels)):
    index = test_labels[row].astype(int)
    test_labels_final[row][index[0]] += 0.8

for i in range(50):
    for row in range(len(train_images)):
        h2 = np.dot(train_images[row][:],weights2)
        
        h3_with_bias[1:] = 1 / ( 1 + np.exp(-h2))
        h4 = np.dot(h3_with_bias,weights3)
        #print h4.shape
        h5 = 1 / ( 1 + np.exp(-h4))
        output_error = h5 * (1 - h5) * (train_labels_final[row][:] - h5) #1*10


        x = h3_with_bias[1:] * ( 1 - h3_with_bias[1:] )
        y =  np.dot(weights3[1:,:],np.transpose(output_error))
        hidden_error = x * y
        #print "X.shape" , x.shape , "y.shape" , y.shape , "hiddem_error", hidden_error.shape

        step_1 = Learning_rate * np.outer(h3_with_bias,output_error)
        step_2 = Momentum * deltaO
        newdeltaO = step_1 + step_2

        step_1 = Learning_rate * np.outer(hidden_error,train_images[row][:])
        #print step_1.shape , hidden_error.shape ,train_images[row][:].shape
        step_2 = Momentum * deltaH
        newdeltaH = step_1 + step_2

        weights3 += newdeltaO
        deltaO = newdeltaO
        weights2 += np.transpose(newdeltaH)
        deltaH = newdeltaH

        index = np.argmax(h5,axis=0)
        finding_accuracy_train[row] =  index


    for row in range(len(test_images)):
        
        h2 = np.dot(test_images[row][:],weights2)

        h3_with_bias[1:] = 1 / ( 1 + np.exp(-h2))
        h4 = np.dot(h3_with_bias,weights3)
        #print h4.shape
        h5 = 1 / ( 1 + np.exp(-h4))

        index = np.argmax(h5,axis=0)
        finding_accuracy_test[row] =  index



cfm_train = confusion_matrix(train_labels,finding_accuracy_train)
diagonal_sum_train =  sum(np.diag(cfm_train))
accuracy_train = (diagonal_sum_train/60000.00)*100
print cfm_train
print accuracy_train


cfm_test = confusion_matrix(test_labels,finding_accuracy_test)
diagonal_sum_test =  sum(np.diag(cfm_test))
accuracy_test = (diagonal_sum_test/10000.00)*100
print cfm_test
print accuracy_test


#print h6
#print train_labels
#cfm = confusion_matrix(h6,train_labels)
#print cfm
#print accuracy
