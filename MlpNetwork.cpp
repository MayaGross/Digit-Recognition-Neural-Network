#include "MlpNetwork.h"


MlpNetwork::MlpNetwork(const Matrix weights[], const Matrix biases[],
    const ActFunc acts[])
{
    for (int i = 0; i < MLP_SIZE; i++)
    {
        _layers[i] = Dense(weights[i], biases[i], acts[i]);
    }
}

digit MlpNetwork::operator()(const Matrix &input) const
{
    Matrix input_copy = input;
    input_copy.vectorize();
    for (int i = 0; i < MLP_SIZE; i++)
    {
        input_copy = _layers[i](input_copy);
    }
    digit result;
    result.value = input_copy.argmax();
    result.probability = input_copy[result.value];
    return  result;
}
