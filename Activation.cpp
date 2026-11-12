#include  "Activation.h"


Matrix activation::relu(const Matrix &m)
{
    Matrix new_mat = m;
    int num_of_elements = (m.get_rows() * m.get_cols());
    for (int i = 0; i < num_of_elements; i++)
    {
        if (new_mat[i] < 0)
        {
            new_mat[i] = 0;
        }
    }
    return new_mat;
}


Matrix activation::softmax(const Matrix &m)
{
    Matrix new_mat = m;
    int num_of_elements = (m.get_rows() * m.get_cols());
    for (int i = 0; i < num_of_elements; i++)
    {
        new_mat[i] = std::exp(new_mat[i]);
    }
    float new_sum = new_mat.sum();
    new_mat = (1/new_sum) * new_mat;
    return new_mat;
}