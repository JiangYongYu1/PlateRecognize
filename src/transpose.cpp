#include "transpose.h"
inline int blob_count(int start_axis, int end_axis, std::vector<int> shape) {
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
        count *= shape[i];
    }
    return count;
}

inline int blob_count(int start_axis, std::vector<int> shape) {
    return blob_count(start_axis, shape.size(), shape);
}

template <typename Dtype>
void Permute(const int count, Dtype *bottom_data, const bool forward,
             const int *permute_order, const int *old_steps, const int *new_steps,
             const int num_axes, Dtype *top_data) {
    for (int i = 0; i < count; ++i) {
        int old_idx = 0;
        int idx = i;
        for (int j = 0; j < num_axes; ++j) {
            int order = permute_order[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }
        if (forward) {
            top_data[i] = bottom_data[old_idx];
        } else {
            bottom_data[old_idx] = top_data[i];
        }
    }
}

template void Permute(const int count, float *bottom_data, const bool forward,
                      const int *permute_order, const int *old_steps, const int *new_steps,
                      const int num_axes, float *top_data);
template void Permute(const int count, double *bottom_data, const bool forward,
                      const int *permute_order, const int *old_steps, const int *new_steps,
                      const int num_axes, double *top_data);

void transpose(float *bottom_data, vector<int> bottom_shape, vector<int> orders, float *top_data, std::vector<int> top_shape) {
    const int top_count = blob_count(0, top_shape);

    const int num_axes_ = bottom_shape.size();
    std::vector<int> permute_order_(num_axes_);
    std::vector<int> old_steps_(num_axes_);
    std::vector<int> new_steps_(num_axes_);
    for (int i = 0; i < num_axes_; ++i) {
        permute_order_[i] = orders[i];
    }

    for (int i = 0; i < num_axes_; ++i) {
        if (i == num_axes_ - 1) {
            old_steps_[i] = 1;
        } else {
            old_steps_[i] = blob_count(i + 1, bottom_shape);
        }
    }

    for (int i = 0; i < num_axes_; ++i) {
        if (i == num_axes_ - 1) {
            new_steps_[i] = 1;
        } else {
            new_steps_[i] = blob_count(i + 1, top_shape);
        }
    }

    bool forward = true;
    Permute(top_count, bottom_data, forward, permute_order_.data(), old_steps_.data(),
            new_steps_.data(), num_axes_, top_data);
}
