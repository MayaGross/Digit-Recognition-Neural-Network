#include "Dense.h"

Dense::Dense(const Matrix &weights, const Matrix &biases, ActFunc act_func):
_weights(weights), _bias(biases), _activation(act_func) {}

Dense::Dense() : Dense(Matrix(), Matrix(), nullptr) {}

Matrix Dense::operator()(const Matrix &m) const
{
    Matrix prod = _weights * m;
    Matrix final_res = prod + _bias;
    return _activation(final_res);
}

const Matrix & Dense::get_weights() const
{
    return _weights;
}

const Matrix & Dense::get_bias() const
{
    return _bias;
}

ActFunc Dense::get_activation() const
{
    return _activation;
}
