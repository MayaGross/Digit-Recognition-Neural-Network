// Activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "Matrix.h"

namespace activation
{
    /**
     * @brief Applies the Relu activation function.
     * Performs element-wise operation: f(x) = max(0, x).
     * Any negative number in the matrix is replaced by 0.
     * @param m The input matrix (const reference).
     * @return A new Matrix with the activation function applied.
     */
    Matrix relu(const Matrix& m);

    /**
     * @brief Applies the Softmax activation function.
     * Transforms the input matrix into a probability distribution.
     * The function computes the exponent of each element and normalizes
     * the vector so that the sum of all elements equals 1.
     * Formula: f(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
     * * @param m The input matrix (const reference).
     * @return A new Matrix where elements are in range [0, 1] and sum to 1.
     */
    Matrix softmax( const Matrix& m);
}
#endif //ACTIVATION_H