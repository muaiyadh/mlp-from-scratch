# Multilayer Perceptron model from scratch in MATLAB
## Dataset:
Download the MNIST dataset to be used with MATLAB from https://lucidar.me/en/matlab/files/mnist.mat

After that, put the `mnist.mat` file in the `Data` folder

## Execution:
1. Open the "Neural_Network_From_Scratch_MNIST.m" file from the "Source Code" folder in MATLAB. Make sure the workspace is in the "Source Code" folder, so the code can find the dataset correctly.

1. You can tweak the Hyperparameters section first (desired accuracy, number of hidden neurons, etc), or use the default values.

1. Now you can run the code in sections if you want to see steps more clearly, or run the entire code.

1. Once you reach the training part, you will see some info in the Console Window, such as the chosen Learning rate, the Epoch number, the Training and Validation accuracies, which can be helpful to tune the hyperparameters and check for any overfitting.

1. Once the code reaches the desired accuracy, or exhausts the maximum number of epochs, you will see the Testing accuracy printed to the Console Window, and two figures. The first figure represents how the Validation accuracy changes over each Epoch, and the second figure shows the Confusion Matrix for the Testing dataset.

## Inputs:
  You can input the Training set size, validation set size, learning rate, maximum epochs count, desired accuracy, and neuron count for each layer (2 hidden layers and 1 output layer)
  
## Outputs:
  Final Testing accuracy in the Console Window, Validation accuracy vs. Epoch # figure, and Confusion Matrix for the Testing dataset.

## Warnings: 
  * The code includes a substitute for some functions that use toolboxes (Statistics and Machine Learning Toolbox), such that the code runs on MATLAB with no additions.
    If necessary, you can comment the new code and uncomment the original code that uses the toolbox.
  * Make sure the size of the Training set and Validation set doesn't exceed the total size of the MNIST training set, which is 60,000 images.
  * If you want to change the datasets to another, you can change the last part of the code that calculates the confusion matrix, as some values there are hardcoded to the MNIST dataset.
