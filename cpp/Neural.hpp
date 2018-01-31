#ifndef Neural_hpp
#define Neural_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <ctime>
#include <limits>

typedef std::vector<size_t> NetworkSize;
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> Data;

class Layer
{
protected:
    size_t inputN, outputN;
    
    Eigen::MatrixXd W, dW;
    Eigen::VectorXd b, db, p, s;
    
    Layer() = default;
    Layer(size_t _inputN, size_t _outputN):
    inputN(_inputN), outputN(_outputN)
    {
        Resize();
    }
    
    void Resize();
    void RandomInitialize(double low_bound, double up_bound);
    
    void ImportFromFile(std::ifstream &infile);
    
    void Forward(Eigen::VectorXd &y, const Eigen::VectorXd &x);
    void Backward(Eigen::VectorXd &sensitivity);
    
    // r is learning rate, beta is L2 regularization strength
    void Update(double r, double beta);
};

class HiddenLayer : private Layer
{
    HiddenLayer() = default;
    HiddenLayer(size_t _inputN, size_t _outputN):
    Layer(_inputN, _outputN) { }
    
    friend class NeuralNetwork;
};

class InputLayer : private Layer
{
public:
    InputLayer() = default;
    InputLayer(size_t _inputN, size_t _outputN):
    Layer(_inputN, _outputN) { }
    
    void Backward();
    
    friend class NeuralNetwork;
};

class OutputLayer : private Layer
{
public:
    OutputLayer() = default;
    OutputLayer(size_t _inputN, size_t _outputN):
    Layer(_inputN, _outputN) { }
    
    void Forward(Eigen::VectorXd &y, const Eigen::VectorXd &x);
    void GetSensitivity(const Eigen::VectorXd &target, const Eigen::VectorXd &result);
    
    friend class NeuralNetwork;
};

class NeuralNetwork
{
    NetworkSize layerSize;
    InputLayer iLayer;
    OutputLayer oLayer;
    std::vector<HiddenLayer> hLayer;

    void ConstructNetworks();
    void RandomInitialize(double low_bound, double up_bound);
    
    void BackPropagate(const Eigen::VectorXd &target, const Eigen::VectorXd &result);
    void SingleForthBack(const Eigen::VectorXd &input, const Eigen::VectorXd &target);
    
    // r is learning rate, beta is L2 regularization strength
    void UpdateNetwork(double r, double beta);

public:
    NeuralNetwork() = default;
    // This constructor creates a new neural network
    NeuralNetwork(const std::vector<size_t> &sizes);
    // This constructor loads a saved neural network
    NeuralNetwork(const std::string &filename);
    
    void ForwardPropagate(Eigen::VectorXd &output, const Eigen::VectorXd &input);

    // training set is a vector of pairs. pair.first is the input, pair.second column is the target
    // r is learning rate, beta is L2 regularization strength
    void TrainNetwork(size_t nEpochs, size_t batch, double r, double beta,
                      const Data &trainingset);
    void TrainNetworkValid(size_t nEpochs, size_t batch, double r, double beta,
                      const Data &trainingset, const Data &validset);
    double TestNetwork(const Data &testset);
    
    void SaveNetwork(const std::string &filename);
    void LoadNetwork(const std::string &filename);
};



#endif /* Neural_hpp */
