// MlpNetwork.h
#ifndef MLPNETWORK_H
#define MLPNETWORK_H
#define MLP_SIZE 4
#include "Dense.h"

//******** Don't touch the next lines! ********
const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
									{64,  128},
									{20,  64},
									{10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
								 {64,  1},
								 {20,  1},
								 {10,  1}};
/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit {
	unsigned int value;
	float probability;
} digit;
//******** From now-on you can write your code ********

class MlpNetwork
{
	Dense _layers[MLP_SIZE];
public:
	MlpNetwork(const Matrix weights[], const Matrix biases[], const ActFunc acts[]);
	digit operator()(const Matrix& input) const;
};


#endif // MLPNETWORK_H