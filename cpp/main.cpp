#include <iostream>
#include "Neural.hpp"
#include "Utility.hpp"

int main(int argc, const char * argv[]) {
    
    //////////////////////////////////////
    // User input data
    NetworkSize size = {2, 20, 10, 1};
    size_t Epoch = 1000;
    size_t BatchSize = 100;
    double LearningRate = 0.01;
    double L2Strength = 0.02;
    size_t TrainingSetSize = 1000;
    size_t ValidationSetSize = 1000;
    size_t TestSetSize = 1000;
    double DataNoise = 0.01;
    //////////////////////////////////////
    
    Data training, validation, test;
    xyData(TrainingSetSize, DataNoise, training);
    xyData(ValidationSetSize, DataNoise, validation);
    xyData(TestSetSize, DataNoise, test);

    NeuralNetwork nn = NeuralNetwork(size);

    nn.TrainNetworkValid(Epoch, BatchSize, LearningRate, L2Strength, training, validation);
    
    std::cout << "MSE error on training set: " << nn.TestNetwork(test) << std::endl;
    
    nn.SaveNetwork("TestNet.nn");
    
    NeuralNetwork load_nn = NeuralNetwork("TestNet.nn");
    std::cout << "MSE error on training set: " << load_nn.TestNetwork(test) << std::endl;

    return 0;
}
