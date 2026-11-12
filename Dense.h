// Dense.h
#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

typedef Matrix (*ActFunc)(const Matrix&);

class Dense
{
    Matrix _weights;
    Matrix _bias;
    ActFunc _activation;

public:
    // Constructors

    /**
     * @brief Constructs a new layer with given weights, bias and activation type.
     * @param weights The weights matrix.
     * @param biases The bias vector (matrix).
     * @param act_func The activation function to use.
     */
    Dense(const Matrix& weights, const Matrix& biases, ActFunc act_func);

    /**
     * @brief Default constructor.
     * Initializes weights and bias with their default constructor,
     * and activation as nullptr.
     */
    Dense();

    // Operators

    /**
     * @brief Applies the layer on the input Matrix.
     * Performs: output = activation(weights * input + bias)
     * @param m The input matrix (usually a vector).
     * @return The output matrix after the layer's operation.
     */
    Matrix operator()(const Matrix& m) const;

    // Getters

    /**
     * @brief Returns the weights of this layer.
     * @return const reference to the weights matrix.
     */
    const Matrix& get_weights() const;

    /**
     * @brief Returns the bias of this layer.
     * @return const reference to the bias matrix.
     */
    const Matrix& get_bias() const;

    /**
     * @brief Returns the activation function pointer of this layer.
     * @return ActFunc pointer.
     */
    ActFunc get_activation() const;
};

#endif //DENSE_H
