import numpy as np
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

Learning_rate = 0.1
Momentum_1 = 0.9
Momentum_2 = 0
Momentum_3 = 0.25
Momentum_4 = 0.5
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

train_labels_test = np.zeros((30000,1))
train_labels_test = train_labels[:30000,:]


train_labels_test_final = np.zeros((30000,10))

train_labels_final = np.full((60000, 10), 0.1, dtype=float)
train_labels_test_final = train_labels_final[:30000,:]


for row in range(len(train_labels_test)):
    index = train_labels_test[row].astype(int)
    train_labels_test_final[row][index[0]] += 0.8


train_images_test = np.zeros((30000,785))
train_images_test = train_images[:30000,:]


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

hidden_layer_weights = np.random.uniform(-0.05, 0.05, size=(785,Hidden_Layer_3))
output_layer_weights_with_bias = np.random.uniform(-0.05, 0.05, size=(Hidden_Layer_3+1,10))

modified_weights_hidden_to_output = np.zeros((Hidden_Layer_3+1,10))
modified_weights_input_to_hidden = np.zeros((Hidden_Layer_3,785))


h6 = np.zeros((30000,1))
finding_accuracy_train = np.zeros((30000,1))
finding_accuracy_test = np.zeros((10000,1))
hj = np.zeros(Hidden_Layer_3+1)
hj[0] = 1

accuracy_train = np.zeros(50)
accuracy_test = np.zeros(50)


for i in range(50):
    for row in range(len(train_images_test)):

        # we take the dot product of the training inputs and the randomly generated weights and then take the sigmoid value of
        # that value to get the actual hidden node value. The same process is applied for the hidden node inputs which are used to 
        # calculate the output node values. The error values for the output and hidden nodes are created and then the weights are
        # changed using these error values. For calculating the accuracy we add the numbers on the diagonals of the training accuracy
        # matrix and the divide it by total number of training examples.

        wjixi = np.dot(train_images_test[row][:],hidden_layer_weights)
        
        hj[1:] = 1 / ( 1 + np.exp(-wjixi))
        wkjhj = np.dot(hj,output_layer_weights_with_bias)
      
        Ok = 1 / ( 1 + np.exp(-wkjhj))
        compute_error_at_output = Ok * (1 - Ok) * (train_labels_final[row][:] - Ok)


        x = hj[1:] * ( 1 - hj[1:] )
        y =  np.dot(output_layer_weights_with_bias[1:,:],np.transpose(compute_error_at_output))
        compute_error_at_hiddenlayer_dj = x * y
        

        step_1 = Learning_rate * np.outer(hj,compute_error_at_output)
        step_2 = Momentum_1 * modified_weights_hidden_to_output
        newmodified_weights_hidden_to_output = step_1 + step_2

        step_1 = Learning_rate * np.outer(compute_error_at_hiddenlayer_dj,train_images_test[row][:])
        
        step_2 = Momentum_1 * modified_weights_input_to_hidden
        newmodified_weights_input_to_hidden = step_1 + step_2

        output_layer_weights_with_bias += newmodified_weights_hidden_to_output
        modified_weights_hidden_to_output = newmodified_weights_hidden_to_output
        hidden_layer_weights += np.transpose(newmodified_weights_input_to_hidden)
        modified_weights_input_to_hidden = newmodified_weights_input_to_hidden

        index = np.argmax(Ok,axis=0)
        finding_accuracy_train[row] =  index


    cfm_train = confusion_matrix(train_labels_test,finding_accuracy_train)
    diagonal_sum_train =  sum(np.diag(cfm_train))
    accuracy_train[i] = (diagonal_sum_train/30000.00)*100


    for row in range(len(test_images)):

        # Wedo the same steps as training the data except we dont have to update the weights. The testing is done on the weights
        # which were updated in the previous training phase
        # For calculating the accuracy of 1 epoch we add the diagonal elements of the testing accuracy matrix and divide it by the
        # total number of elements we are considering for testing.
        
        wjixi = np.dot(test_images[row][:],hidden_layer_weights)

        hj[1:] = 1 / ( 1 + np.exp(-wjixi))
        wkjhj = np.dot(hj,output_layer_weights_with_bias)
      
        Ok = 1 / ( 1 + np.exp(-wkjhj))

        index = np.argmax(Ok,axis=0)
        finding_accuracy_test[row] =  index

    cfm_test = confusion_matrix(test_labels,finding_accuracy_test)
    diagonal_sum_test =  sum(np.diag(cfm_test))
    accuracy_test[i] = (diagonal_sum_test/10000.00)*100



plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.ylabel("Accuracy in %")
plt.xlabel("Epoch")

image= "quarterexamples.png"
plt.title("For 100 hidden units and Momentum value of 0.9 using one quarter of training examples")
plt.savefig(image)
plt.show()
print "CONFUSION MATRIX OF TRAIN SET : For 100 hidden units and Momentum value of 0.9 using one quarter of training examples \n"
print cfm_train
print "\n"

print "CONFUSION MATRIX OF TEST SET : For 100 hidden units and Momentum value of 0.9 using one quarter of training examples\n"
print cfm_test

print "\n"
print "accuracy_train: "
print accuracy_train

print "\n"
print "accuracy_test: "
print accuracy_test

MLP_final_test.py
Displaying MLP_final_test.py.