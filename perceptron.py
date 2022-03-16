##PERCEPTRON

import numpy as np
theta = 1                        
epoch = 3                      
 
class Perceptron(object):
    def __init__(self, input_size, learning_rate=1):
        self.learning_rate = learning_rate
        self.weights = [int(0) for i in range(input_size+1)]
        print("weight - {}".format(self.weights))
           
    def predict(self, x):
        
        ans = self.weights[1] * x[0] + self.weights[2] * x[1]
        return self.weights[0] + ans
                
    def train(self, x, y, weights):           
        for inputs, label in zip(x, y):
            net_in = self.predict(inputs)
            if net_in > theta:                        
                y_out = 1
            elif net_in < -theta:
                y_out = -1
            else:
                y_out = 0
            if y_out != label: # updating the net on incorrect prediction                    
                self.weights[1] += self.learning_rate * label * inputs[0] # W = alpha * Y * X
                self.weights[2] += self.learning_rate * label * inputs[1]
                self.weights[0] += self.learning_rate * label  # B = alpha * Y         
            print(inputs, net_in, label, y_out, self.weights)
 
if __name__ == "__main__":
    x = []                    
    x.append([1, 1])  
    x.append([1, -1])
    x.append([-1, 1])
    x.append([-1, -1])
 
    y = [1, -1, -1, -1]          
 
    perceptron = Perceptron(2,1)           
 
    for i in range(epoch):
        print("Epoch",i)
        print("X1 X2 ", " Net ", " T ", " Y ", " B Weights")    
        weights = perceptron.weights
        print("Initial Weights", weights)
        perceptron.train(x, y, weights)
   
        
##PERCEPTRON 2:
   
import numpy as np
teta = 1                         #Threshold
epoch = 3                        #Epoch value, change to repeat the learining till network converge

class Perceptron(object):
    
    def __init__(self, no_of_inputs, learning_rate=0.2 ):
        self.learning_rate = learning_rate
        self.weights =  np.zeros(no_of_inputs + 1)     #Initialize weights & bias as zero, weights[0] is used as bias
           
    def predict(self, inputs):
         return (np.dot(inputs, self.weights[1:]) + self.weights[0])  # Returns the net_in
                
    def train(self, training_inputs, t,weights):             #Calculate the weights for each input
        for inputs, label in zip(training_inputs, t):
            net_in = self.predict(inputs)
            if net_in > teta:                        #Find the activation output Y
                y_out = 1
            elif net_in < -teta:
                y_out = -1
            else:
                y_out = 0
            if y_out != label:                     #If t != y , update the weights, otherwise W(new)=W(old)
                self.weights[1:] += self.learning_rate * label * inputs #W(new)
                self.weights[0] += self.learning_rate * label           #b(new)
            print(inputs, net_in, label, y_out,self.weights)
    def disp(self):
        print(self.weights)

training_inputs = []                    #Holds vector of inputs for AND/OR logic
training_inputs.append(np.array([1, 1]))  
training_inputs.append(np.array([1, -1]))
training_inputs.append(np.array([-1, 1]))
training_inputs.append(np.array([-1, -1]))

t = np.array([1, 1, 1, -1])          #Target output

perceptron = Perceptron(2)            #Create 2 input object vector

for i in range(epoch):
    print("Epoch",i)
    print("X1 X2","Net"," T"," Y", "B Weights")     # For output display pattern
    weights = perceptron.weights
    print("Initial Weights",weights)
    perceptron.train(training_inputs, t,weights) 
    
    
 
