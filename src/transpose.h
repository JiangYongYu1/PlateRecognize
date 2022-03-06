#pragma once
#include <vector>
using std::vector;

void transpose(float *bottom_data, vector<int> bottom_shape, 
               vector<int> orders, float *top_data, 
               std::vector<int> top_shape);