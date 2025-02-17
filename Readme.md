# Building a Convolution Neural Network

I know I am a little late to the party but at least I am starting :)

## Objective

Build a convolutional neural network builder with the focus on classification but should essentially be applicable
to other categories of problems.

## Scope

While the idea is to come up with a fairly generic (almost library/framework like) CNN builder,
the primary goal is to understand the fundamental elements and the workings of a CNN
and not to write great code(pretty sure LLMs running on a Raspi will beat me there).

## Background

### What is a neural network?

Bunch of "connected" layers of neurons with weights and biases that take in an input and present an output;
Essentially like any other function but what makes them special is the ability to bake non-linearity to it;
Additionally, while I did say "an output", the output is typically a list of probabilities of different outputs(in case of classifiers);

### Elements of a neural network

- Neuron: The fundamental data structure made up of weights and biases.
= Layer: A collection of neurons; a network has at least one input layer, one hidden(middle) layer and one output layer (this arrangement is also called perceptron)
- Weight: A _typically_ floating point value assigned to a neuron.
- Bias: Noise correction (to compensate for the noise in the input) added to neurons.
- Activation fn: Function applied at each neuron; args include: input, weight and bias; output: discrete or continuous value signifying the "contribution" of the neuron towards the overall output of the network.

### Training a neural network

Classification is typically a supervised technique; You are expected to have labeled data to train the network;
Validation dataset is used to test the accuracy of the network.

- The most prevalent and efficient technique is **Back Propagation**.
- The idea is to start with a network (to keep it simple, let's stick with perceptron) with neurons with _arbitrary_ weights and biases.
- The model _architecture_(number of neurons per layer, connectivity between layers) depends on the type of problem you are solving:
  - If you are solving an image classification problem, the input layer will likely consist of N neurons with N being the number of pixels in the image; the number of neurons at the output layer would be equal to the number of categories of images the model should be able to classify; the number of neurons in the hidden layer can be artibrary.
- For each image from the test data, _encode_ the data to match the format expected by the input layer.
  - For an image classifier this could be the greyscale value of each pixel in the image.
- We then being _forward propagation_ where the inputs are propagated to the neurons in the hidden layer.
- At each neuron, we apply the activation function and propagate the result to the next layer and so on, until the ouput layer.
  - To note is the the function applied at the output layer should match the desired outcome; we can apply a non-linear function at the hidden layers (e.g. ReLU) but can either apply no function(regression) or binary function(e.g. Sigmoid) or other function at the output layer.
- So far what we have done is _inference_; this is what a trained model would also do for a given input.
- When it comes to training, we will have to evaluate the output and feed back the evaluation back to the network(model) for it to correct/improve its weights and biases.
  - When we say "evaluate the output", it means we are checking the accuracy of the model.
- This process is called Back Propagation; The most common way is **Gradient Descent**.
- TODO: Gradient descent explain

## Design

### Binary classifier

We will start with a neural network generator that can be used a binary classifier.

Input: TODO

Output: Yes/No

Metrics:
There are various metrics in a binary classification(_True Positive=TP,True Negative=TN,False Positive=FP,False Negative=FN_):

- Accuracy: How many accurate classifications can the model make (Accuracy = TP+TN/(TP+FP+TN+FN))
- Precision: How precise is the model at positive predictions (Precision = TP/TP+FP)
- Recall: How good is the model at NOT misclassifying positive results (Recall = TP/TP+FN)
- False Positive Rate: FPR = FP/FP+TN
