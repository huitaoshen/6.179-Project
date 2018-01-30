#include "Neural.hpp"

using namespace Eigen;

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &sizes):
layerSize(sizes)
{
    ConstructNetworks();
    
    RandomInitialize(-1., 1.);
}

NeuralNetwork::NeuralNetwork(const std::string &filename)
{
    LoadNetwork(filename);
}

void NeuralNetwork::ConstructNetworks()
{
    assert(layerSize.size() > 2);
    
    iLayer = InputLayer(layerSize[0], layerSize[1]);
    
    hLayer.clear();
    for (int i = 1; i < layerSize.size() - 2; ++i) {
        hLayer.push_back(HiddenLayer(layerSize[i], layerSize[i + 1]));
    }
    
    oLayer = OutputLayer(layerSize[layerSize.size() - 2], layerSize[layerSize.size() - 1]);
}

void NeuralNetwork::RandomInitialize(double low_bound, double up_bound)
{
    srand(static_cast <unsigned> (time(0)));
    
    iLayer.RandomInitialize(low_bound, up_bound);
    for (int i = 0; i < hLayer.size(); ++i) {
        hLayer[i].RandomInitialize(low_bound, up_bound);
    }
    oLayer.RandomInitialize(low_bound, up_bound);
    
}

void NeuralNetwork::ForwardPropagate(VectorXd &output, const VectorXd &input)
{
    assert(input.size() == iLayer.inputN);
    
    VectorXd tmp = input;
    iLayer.Forward(output, tmp);
    tmp = output;
    
    for (int i = 0; i < hLayer.size(); ++i) {
        hLayer[i].Forward(output, tmp);
        tmp = output;
    }
    
    oLayer.Forward(output, tmp);
}

void NeuralNetwork::BackPropagate(const Eigen::VectorXd &target, const Eigen::VectorXd &result)
{
    oLayer.GetSensitivity(target, result);
    
    if (hLayer.size() > 0) {
        oLayer.Backward(hLayer[hLayer.size() - 1].s);
        
        for (int i = hLayer.size() - 1; i > 0; --i) {
            hLayer[i].Backward(hLayer[i - 1].s);
        }
        hLayer[0].Backward(iLayer.s);
    } else {
        oLayer.Backward(iLayer.s);
    }
    
    iLayer.Backward();
}

void NeuralNetwork::UpdateNetwork(double r, double beta)
{
    assert(r > 0 && beta >= 0);
    oLayer.Update(r, beta);
    for (int i = 0; i < hLayer.size(); ++i) {
        hLayer[i].Update(r, beta);
    }
    iLayer.Update(r, beta);
}

void NeuralNetwork::SingleForthBack(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    VectorXd result;
    
    ForwardPropagate(result, input);
    BackPropagate(target, result);
}

void NeuralNetwork::TrainNetwork(size_t nEpochs, size_t batch, double r, double beta, const Data &trainingset)
{
    if (batch == 0)
        batch = trainingset.size();
    
    size_t cbatch = 0;
    
    for (int i = 0; i < nEpochs; ++i) {
        std::cout << "* Epoch " << i + 1 << " (Batch size = " << batch << ") *" << std::endl;
        for (int j = 0; j < trainingset.size(); ++j) {
            SingleForthBack(trainingset[j].first, trainingset[j].second);
            ++cbatch;
            
            if (cbatch == batch) {
                UpdateNetwork(r / (double)cbatch, beta);
                cbatch = 0;
            }
        }
    }
}

void NeuralNetwork::TrainNetworkValid(size_t nEpochs, size_t batch, double r, double beta, const Data &trainingset, const Data &validset)
{
    if (batch == 0)
        batch = trainingset.size();
    
    size_t cbatch = 0;
    
    for (int i = 0; i < nEpochs; ++i) {
        std::cout << "* Epoch " << i + 1 << " (Batch size = " << batch << ") *" << std::endl;
        for (int j = 0; j < trainingset.size(); ++j) {
            SingleForthBack(trainingset[j].first, trainingset[j].second);
            ++cbatch;
            
            if (cbatch == batch) {
                UpdateNetwork(r / (double)cbatch, beta);
                cbatch = 0;
            }
        }
        std::cout << "  Validation loss: " << TestNetwork(validset) << std::endl;
    }
}

double NeuralNetwork::TestNetwork(const Data &data)
{
    double mse = 0;
    VectorXd res, loss;
    for (auto i = 0; i < data.size(); ++i) {
        ForwardPropagate(res, data[i].first);
        loss = res - data[i].second;
        mse += loss.norm();
    }
    return mse / (double) data.size();
}

void NeuralNetwork::SaveNetwork(const std::string &filename)
{
    std::cout << "Saving neural network to " << filename << "... ";
    
    std::ofstream inFile;
    inFile.precision(std::numeric_limits<double>::max_digits10);
    
    if (filename.size() > 0) {
        inFile.open(filename, std::ios::trunc);
        for (int i = 0; i < layerSize.size(); ++i) {
            inFile << layerSize[i] << "\t";
        }
        inFile << std::endl;
        inFile << iLayer.W << std::endl << iLayer.b << std::endl;
        for (int i = 0; i < hLayer.size(); ++i) {
            inFile << hLayer[i].W << std::endl << hLayer[i].b << std::endl;
        }
        inFile << oLayer.W << std::endl << oLayer.b << std::endl;
        inFile.close();
    }
    
    std::cout << "completed. " << std::endl;
}

void NeuralNetwork::LoadNetwork(const std::string &filename)
{
    std::vector<std::vector<double>> data;
    std::ifstream infile(filename);
    
    assert(infile);
    
    std::cout << "Importing nerual network from " << filename << "... ";
    std::string line;
    while (getline(infile, line))
    {
        std::istringstream iss(line);
        data.push_back(std::vector<double>(std::istream_iterator<double>(iss),
                                           std::istream_iterator<double>()));
    }
    assert(data.size() > 0);
    
    layerSize.resize(data[0].size());
    for (int i = 0; i < data[0].size(); ++i) {
        layerSize[i] = data[0][i];
    }
    
    ConstructNetworks();
    
    size_t row_flag = 1;
    
    for (int i = 0; i < iLayer.W.rows(); ++i) {
        assert(data[row_flag].size() == iLayer.W.cols());
        iLayer.W.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
        data[row_flag].clear();
        ++row_flag;
    }
    for (int i = 0; i < iLayer.b.rows(); ++i) {
        assert(data[row_flag].size() == iLayer.b.cols());
        iLayer.b.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
        data[row_flag].clear();
        ++row_flag;
    }
    for (int k = 0; k < hLayer.size(); ++k) {
        for (int i = 0; i < hLayer[k].W.rows(); ++i) {
            assert(data[row_flag].size() == hLayer[k].W.cols());
            hLayer[k].W.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
            data[row_flag].clear();
            ++row_flag;
        }
        for (int i = 0; i < hLayer[k].b.rows(); ++i) {
            assert(data[row_flag].size() == hLayer[k].b.cols());
            hLayer[k].b.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
            data[row_flag].clear();
            ++row_flag;
        }
    }
    for (int i = 0; i < oLayer.W.rows(); ++i) {
        assert(data[row_flag].size() == oLayer.W.cols());
        oLayer.W.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
        data[row_flag].clear();
        ++row_flag;
    }
    for (int i = 0; i < oLayer.b.rows(); ++i) {
        assert(data[row_flag].size() == oLayer.b.cols());
        oLayer.b.row(i) = VectorXd::Map(&data[row_flag][0], data[row_flag].size());
        data[row_flag].clear();
        ++row_flag;
    }
    
    std::cout << "completed. " << std::endl;
}

void Layer::Resize()
{
    W.resize(outputN, inputN);
    b.resize(outputN);
    dW = MatrixXd::Zero(outputN, inputN);
    db = VectorXd::Zero(outputN);
    
}

void Layer::RandomInitialize(double low_bound, double up_bound)
{
    // Initialize both weight and bias using random numbers uniformally distributed between
    // low_bound and up_bound
    
    assert(low_bound < up_bound);
    
    for (int i = 0; i < outputN; ++i) {
        for (int j = 0; j < inputN; ++j) {
            W(i, j) = low_bound + static_cast<double> (rand()) / (static_cast<double> (RAND_MAX / (up_bound - low_bound)));
        }
        b(i) = low_bound + static_cast<double> (rand()) / (static_cast<double> (RAND_MAX / (up_bound - low_bound)));
    }
}

void Layer::Forward(VectorXd &y, const VectorXd &x)
{
    p = x;
    y = W * x + b;
    
    // sigmoid
    // maybe implement by Eigen bitwise function later
    for (int i = 0; i < outputN; ++i) {
        y(i) = 1 / (1 + exp(-y(i)));
    }
}

void Layer::Backward(VectorXd &sensitivity)
{
    dW += s * p.transpose();
    db += s;
    
    sensitivity = W.transpose() * s;
    
    for (int i = 0; i < sensitivity.size(); ++i) {
        sensitivity(i) *= p(i) * (1 - p(i));
    }
}

void Layer::Update(double r, double beta)
{
    W -= r * (dW + beta * W);
    
    b -= r * db;
    
    dW = MatrixXd::Zero(outputN, inputN);
    db = VectorXd::Zero(outputN);
}

void InputLayer::Backward()
{
    dW += s * p.transpose();
    db += s;
}

void OutputLayer::Forward(VectorXd &y, const VectorXd &x)
{
    p = x;
    y = W * x + b;
}

void OutputLayer::GetSensitivity(const Eigen::VectorXd &target, const Eigen::VectorXd &result)
{
    assert(target.size() == result.size());
    s = -2 * (target - result);
}
