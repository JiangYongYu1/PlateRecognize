#pragma once
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
class BlobData {
  public:
    BlobData() : data_(NULL), count_(0) {}

    /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
    BlobData(const int num, const int channels, const int height, const int width) {
        data_ = NULL;
        Reshape(num, channels, height, width);
    }
    BlobData(const std::vector<int> &shape) {
        data_ = NULL;
        Reshape(shape);
    }

    ~BlobData() {
        free_data();
    }
    inline void Reshape(const int num, const int channels, const int height, const int width) {
        std::vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }

    inline void Reshape(const std::vector<int> &shape) {
        count_ = 1;
        shape_.resize(shape.size());

        for (size_t i = 0; i < shape.size(); ++i) {
            count_ *= shape[i];
            shape_[i] = shape[i];
        }
        free_data();
        data_ = (float *)malloc(count_ * sizeof(float));
    }
    inline const std::vector<int> &shape() const {
        return shape_;
    }
    /**
      * @brief Returns the dimension of the index-th axis (or the negative index-th
      *        axis from the end, if index is negative).
      *
      * @param index the axis index, which may be negative as it will be
      *        "canonicalized" using CanonicalAxisIndex.
      *        Dies on out of range index.
      */
    inline int shape(int index) const {
        return shape_[CanonicalAxisIndex(index)];
    }
    inline int num_axes() const {
        return shape_.size();
    }
    inline int count() const {
        return count_;
    }

    /**
      * @brief Compute the volume of a slice; i.e., the product of dimensions
      *        among a range of axes.
      *
      * @param start_axis The first axis to include in the slice.
      *
      * @param end_axis The first axis to exclude from the slice.
      */
    inline int count(int start_axis, int end_axis) const {
        int count = 1;
        for (int i = start_axis; i < end_axis; ++i) {
            count *= shape(i);
        }
        return count;
    }
    /**
      * @brief Compute the volume of a slice spanning from a particular first
      *        axis to the final axis.
      *
      * @param start_axis The first axis to include in the slice.
      */
    inline int count(int start_axis) const {
        return count(start_axis, num_axes());
    }

    /**
      * @brief Returns the 'canonical' version of a (usually) user-specified axis,
      *        allowing for negative indexing (e.g., -1 for the last axis).
      *
      * @param axis_index the axis index.
      *        If 0 <= index < num_axes(), return index.
      *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
      *        e.g., the last axis index (num_axes() - 1) if index == -1,
      *        the second to last if index == -2, etc.
      *        Dies on out of range index.
      */
    inline int CanonicalAxisIndex(int axis_index) const {
        if (axis_index < 0) {
            return axis_index + num_axes();
        }
        return axis_index;
    }

    /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
    inline int num() const {
        return LegacyShape(0);
    }
    /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
    inline int channels() const {
        return LegacyShape(1);
    }
    /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
    inline int height() const {
        return LegacyShape(2);
    }
    /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
    inline int width() const {
        return LegacyShape(3);
    }
    inline int LegacyShape(int index) const {
        if (index >= num_axes() || index < -num_axes()) {
            // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
            // indexing) -- this special case simulates the one-padding used to fill
            // extraneous axes of legacy blobs.
            return 1;
        }
        return shape(index);
    }

    inline int offset(const int n, const int c = 0, const int h = 0,
                      const int w = 0) const {
        return ((n * channels() + c) * height() + h) * width() + w;
    }

    inline int offset(const std::vector<int> &indices) const {
        int offset = 0;
        for (int i = 0; i < num_axes(); ++i) {
            offset *= shape(i);
            if ((int)indices.size() > i) {
                offset += indices[i];
            }
        }
        return offset;
    } 
    inline float *data() {
        return data_;
    }

  private:
    inline void free_data() {
        if (data_ != NULL) {
            free(data_);
            data_ = NULL;
        }
    }

  protected:
    std::vector<int> shape_;
    float *data_;
    int count_;
};