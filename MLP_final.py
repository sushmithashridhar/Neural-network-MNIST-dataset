import numpy as np
import gzip
from sklearn.metrics import confusion_matrix

Learning_rate = 0.1
Momentum = 0.9
Hidden_Layer_1 = 20
Hidden_Layer_2 = 50
Hidden_Layer_3 = 100


########################################################## TRAIN ################################################################
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

############################################################# TEST ################################################################

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

####################################################################################################################################

hidden_layer_weights = np.random.uniform(-0.05, 0.05, size=(785,Hidden_Layer_1))
output_layer_weights_with_bias = np.random.uniform(-0.05, 0.05, size=(Hidden_Layer_1+1,10))

modified_weights_hidden_to_output = np.zeros((Hidden_Layer_1+1,10))
modified_weights_input_to_hidden = np.zeros((Hidden_Layer_1,785))


h6 = np.zeros((60000,1))
finding_accuracy_train = np.zeros((60000,1))
finding_accuracy_test = np.zeros((10000,1))
hj = np.zeros(Hidden_Layer_1+1)
hj[0] = 1



for i in range(1):
    for row in range(len(train_images)):
        wjixi = np.dot(train_images[row][:],hidden_layer_weights)
        
        hj[1:] = 1 / ( 1 + np.exp(-wjixi))
        wkjhj = np.dot(hj,output_layer_weights_with_bias)
        #print wkjhj.shape
        Ok = 1 / ( 1 + np.exp(-wkjhj))
        compute_error_at_output = Ok * (1 - Ok) * (train_labels_final[row][:] - Ok) #1*10


        x = hj[1:] * ( 1 - hj[1:] )
        y =  np.dot(output_layer_weights_with_bias[1:,:],np.transpose(compute_error_at_output))
        compute_error_at_hiddenlayer_dj = x * y
        #print "X.shape" , x.shape , "y.shape" , y.shape , "hiddem_error", compute_error_at_hiddenlayer_dj.shape

        step_1 = Learning_rate * np.outer(hj,compute_error_at_output)
        step_2 = Momentum * modified_weights_hidden_to_output
        newmodified_weights_hidden_to_output = step_1 + step_2

        step_1 = Learning_rate * np.outer(compute_error_at_hiddenlayer_dj,train_images[row][:])
        #print step_1.shape , compute_error_at_hiddenlayer_dj.shape ,train_images[row][:].shape
        step_2 = Momentum * modified_weights_input_to_hidden
        newmodified_weights_input_to_hidden = step_1 + step_2

        output_layer_weights_with_bias += newmodified_weights_hidden_to_output
        modified_weights_hidden_to_output = newmodified_weights_hidden_to_output
        hidden_layer_weights += np.transpose(newmodified_weights_input_to_hidden)
        modified_weights_input_to_hidden = newmodified_weights_input_to_hidden

        index = np.argmax(Ok,axis=0)
        finding_accuracy_train[row] =  index


    for row in range(len(test_images)):
        
        wjixi = np.dot(test_images[row][:],hidden_layer_weights)

        hj[1:] = 1 / ( 1 + np.exp(-wjixi))
        wkjhj = np.dot(hj,output_layer_weights_with_bias)
        #print wkjhj.shape
        Ok = 1 / ( 1 + np.exp(-wkjhj))

        index = np.argmax(Ok,axis=0)
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
