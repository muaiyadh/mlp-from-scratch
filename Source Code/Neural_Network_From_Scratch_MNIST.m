clc
clear
close all

load("../Data/mnist.mat") % load the MNIST dataset
x_orig = training.images;
x_orig = reshape(x_orig, [28*28, size(x_orig, 3)]).';

% This function is from the Statistical & Machine learning toolbox.
% The code below gives the same output, but works even without the toolbox.
%d_orig = dummyvar(categorical(training.labels));

%%%%%
% Encoding labels without statistical & machine learning toolbox
% Assume 'training.labels' is your label vector
labels = training.labels;
numSamples = length(labels);

% Initialize an all-zero matrix for one-hot encoded labels
d_orig = zeros(numSamples, 10);

% Fill in ones at appropriate positions
for i = 1:numSamples
    d_orig(i, labels(i)+1) = 1;     % change the 0 at the label's position+1 to 1
end
%%%%%


% Get number of samples
n_samples = size(d_orig,1);

% Generate random permutation of indices
idx = randperm(n_samples);

% Shuffle inputs and outputs (helps with training if the data is sorted)
x_orig = x_orig(idx,:);
d_orig = d_orig(idx,:);


%% Initialize the network hyperparameters
training_set_size = 50000;   % # of samples to use in training set
validation_set_size = 10000; % # of samples to use in validation set
eta = 0.1;                   % learning rate
max_epoch=200;               % maximum epoch number
desired_acc=97;              % desired accuracy in percentage
hn1=70;                      % hidden layer 1 neurons count
hn2=70;                      % hidden layer 2 neurons count
on=10;                       % output layer neurons count


% Weights
W1=0.01*randn(size(x_orig,2),hn1);
W2=0.01*randn(hn1,hn2);
W3=0.01*randn(hn2,on);

% Biases
B1=0.01*randn(1,hn1);
B2=0.01*randn(1,hn2);
B3=0.01*randn(1,on);

%% Inputs and Outputs of the NN

% Training set
x = x_orig(1:training_set_size,:);
[m, n] = size(x);
d = d_orig(1:training_set_size,:);

% Validation set
x_v = x_orig(training_set_size:training_set_size+validation_set_size,:);
[m_v, n_v] = size(x_v);
d_v = d_orig(training_set_size:training_set_size+validation_set_size,:);

%% Training
accuracies = zeros(1,max_epoch);
for i = 1:max_epoch
    train_acc = 0;
    train_predicted_labels = zeros(m,on);
    
    valid_acc = 0;
    validation_predicted_labels = zeros(m,on);
    
    for j = 1:m
        X = x(j,:);		% Get the J'th sample

        % Forward propagation
        % 1st Layer
        S1 = X*W1 + B1;
        H1 = 1 ./ (1 + exp(-S1));
        
        % 2nd Layer
        S2 = H1*W2 + B2;
        H2 = 1 ./ (1 + exp(-S2));
        
        % 3rd (output) Layer
        S3 = H2*W3 + B3;
        Y = 1 ./ (1 + exp(-S3));
        
        [~, max_idx] = max(Y);
        train_predicted_labels(j, max_idx) = 1;
    
        T = d(j, :);	% Label of the sample
        
        % Backpropagation
        dEdY  = -(T-Y);
        dEdH2 = (dEdY .* (Y  .*(1-Y)))  * W3';
        dEdH1 = (dEdH2.* (H2 .*(1-H2))) * W2';
        
        dEdW3 = H2' * dEdY  .* (Y  .*(1-Y));
        dEdW2 = H1' * dEdH2 .* (H2 .*(1-H2));
        dEdW1 =  X' * dEdH1 .* (H1 .*(1-H1));

        dEdB3 = dEdY  .* (Y  .*(1-Y));
        dEdB2 = dEdH2 .* (H2 .*(1-H2));
        dEdB1 = dEdH1 .* (H1 .*(1-H1));

		% Update weights and biases
        W3 = W3 - eta .* dEdW3;
        W2 = W2 - eta .* dEdW2;
        W1 = W1 - eta .* dEdW1;

        B3 = B3 - eta .* dEdB3;
        B2 = B2 - eta .* dEdB2;
        B1 = B1 - eta .* dEdB1;
        
        if isequal(T, train_predicted_labels(j,:))
            train_acc = train_acc + 1;
        end
    end
    
    % Convert train_acc to percentage
    train_acc = train_acc/m * 100;
    

    % Check accuracy on Validation set as well. More accurate representation of accuracy than train_acc
    % and helps detecting overfitting.
    for j = 1:m_v
        X=x_v(j,:);                % Apply sample J as input to the network (1,784)
        S1 = X*W1 + B1;            % (1,784) x (784,hn1) = (1,hn1)
        H1 = 1 ./ (1+exp(-S1));    % Output of the 1st hidden layer

        S2 = H1*W2 + B2;           % (1,hn1) x (hn1,hn2) = (1,hn2)
        H2 = 1./ (1+exp(-S2));     % Output of the 2nd hidden layer

        S3 = H2*W3 + B3;           % (1,hn2) x (hn2,on) = (1,on)
        O = 1 ./ (1 + exp(-S3));   % Output of the neural network
        
        [M, M_idx] = max(O);

        validation_predicted_labels(j, M_idx) = 1;        % Method to limit output layer to only one output
        
        T = d_v(j,:);             						  % Label for current sample
        if isequal(T,validation_predicted_labels(j,:))    % Increment accuracy
            valid_acc = valid_acc + 1;
        end
    end

    valid_acc = valid_acc / m_v *100;
    %disp("Learning rate: " + eta + "\tEpoch: " + i + "\tTraining Acc: " + train_acc + "%\tValidation Acc: " + valid_acc + "%");
    % Print out the data
    fprintf("Learning rate: %.2f  Epoch: %d  Training Acc: %.2f%%   Validation Acc: %.2f%%\n", eta, i, train_acc, valid_acc);
    
    accuracies(i) = valid_acc;
    if (valid_acc >= desired_acc)
        disp("Reached desired accuracy!");
        break
    end
end

%% Testing

xt = test.images;
xt = reshape(xt, [28*28, size(xt, 3)]).';
dt = dummyvar(categorical(test.labels));

% Get number of samples
nt_samples = size(dt,1);

% Generate random permutation of indices
tidx = randperm(nt_samples);

% Shuffle inputs and outputs
xt = xt(tidx,:);
dt = dt(tidx,:);

[mt,nt] = size(xt);

test_acc = 0;
test_predicted_labels = zeros(mt,on);
for j=1:mt
    X=xt(j,:);                 % Apply sample J as input to the network (1,784)
    S1 = X*W1 + B1;            % (1,784) x (784,hn1) = (1,hn1)
    H1 = 1 ./ (1+exp(-S1));    % Output of the hidden layer 1

    S2 = H1*W2 + B2;           % (1,hn1) x (hn1,hn2) = (1,hn2)
    H2 = 1./ (1+exp(-S2));     % Output of the hidden layer 2    

    S3 = H2*W3 + B3;           % (1,hn2) x (hn2,on) = (1,on)
    Y = 1 ./ (1 + exp(-S3));   % Output of neural network
    [~, M_idx] = max(Y);

    test_predicted_labels(j,M_idx) = 1;        % Method to limit output layer to only one output
    
    T = dt(j,:);               % Label for current sample

    if isequal(T,test_predicted_labels(j,:))
        test_acc = test_acc+1;
    end
end
test_acc = test_acc/mt * 100;
disp("Testing accuracy: " + test_acc + "%");

%% Visualizing results

%%%
% Validation accuracy plot
%%%

% Plot validation accuracy vs epochs to see if any degradation happened (helps detecting overfitting)
figure
plot(accuracies(accuracies>0))
hold on, grid on
plot(accuracies(accuracies>0), 'b*')
title("Validation accuracy")
ylabel("Accuracy %")
xlabel("Epochs")
hold off

%%%
% Confusion matrix
%%%
numSamples = size(dt, 1);

% Initialize an all-zero vector for decoded labels
train_labels_decoded = zeros(numSamples, 1);
train_predicted_labels_decoded = zeros(numSamples, 1);

% Retrieve the index of ones in each row of d_orig
for i = 1:numSamples
    train_predicted_labels_decoded(i) = find(test_predicted_labels(i, :) == 1) - 1; % subtract 1 to get back original label
    train_labels_decoded(i) = find(dt(i, :) == 1) - 1; % subtract 1 to get back original label
end

% Compute the confusion matrix
cm = confusionmat(train_labels_decoded, train_predicted_labels_decoded);

% Visualize the confusion matrix
figure;
imagesc(cm);

% Display color bar
colorbar;

% Specify the axis labels and title
xlabel('Predicted Labels');
ylabel('True Labels');
title('Confusion Matrix');

% Change the tick marks and labels
xticks(1:10);
yticks(1:10);
xticklabels(0:9);
yticklabels(0:9);
