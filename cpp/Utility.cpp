//
//  Utility.cpp
//  6.179Final
//
//  Created by Andrew Shen on 1/29/18.
//  Copyright © 2018 Andrew Shen. All rights reserved.
//

#include "Utility.hpp"

using namespace Eigen;

void SinData(size_t n, double noi, Data &data) {
    // generate points of size n from f(x)=sin(x) function with tunable random noise
    data.clear();
    
    VectorXd noise = VectorXd::Random(n);
    
    VectorXd x, y;
    x.resize(1);
    y.resize(1);
    
    for (auto i = 0; i < n; ++i) {
        std::vector<VectorXd> vec;
        x(0) = static_cast<double> (rand()) / (static_cast<double> (RAND_MAX / 2 * M_PI));
        y(0) = sin(x(0)) + noi * noise(i);
        data.push_back(std::make_pair(x, y));
    }
}

void xyData(size_t n, double noi, Data &data) {
    // generate points of size n from a f(x,y)=x y function with tunable random noise
    data.clear();
    
    VectorXd noise = VectorXd::Random(n);
    
    VectorXd x, y;
    x.resize(2);
    y.resize(1);
    
    for (auto i = 0; i < n; ++i) {
        std::vector<VectorXd> vec;
        x(0) = static_cast<double> (rand()) / (static_cast<double> (RAND_MAX));
        x(1) = static_cast<double> (rand()) / (static_cast<double> (RAND_MAX));
        y(0) = x(0) * x(1) + noi * noise(i);
        data.push_back(std::make_pair(x, y));
    }
}

