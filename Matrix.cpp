#include "Matrix.h"



Matrix::Matrix(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::length_error("rows and cols must be positive ints.");
    }
    _rows = rows;
    _cols = cols;
    _data = new float[rows * cols]();

}

Matrix::Matrix() : Matrix(1,1) {}

Matrix::Matrix(const Matrix &other) : _rows(other._rows), _cols(other._cols)
{
    _data = new float[_rows * _cols];
    for (int i = 0; i < _rows * _cols; i++)
    {
        _data[i] = other._data[i];
    }
}

Matrix::~Matrix()
{
    delete[] _data;
}

Matrix & Matrix::operator=(const Matrix &other)
{
    if(this == &other)
    {
        return *this;
    }
    float* temp = new float[other._rows * other._cols];
    for (int i = 0; i < other._rows * other._cols; i++)
    {
        temp[i] = other._data[i];
    }
    delete[] _data;
    _data = temp;
    _rows = other._rows;
    _cols = other._cols;
    return *this;
}

Matrix & Matrix::operator+=(const Matrix &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::length_error("dimensions do not match");
    }
    for (int i = 0; i < _rows * _cols; i++)
    {
        _data[i] += other._data[i];
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix new_mat = *this;
    new_mat += other;
    return new_mat;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (_cols != other._rows)
    {
        throw std::length_error("dimensions do not match");
    }

    Matrix new_mat(_rows, other._cols);
    // Iterate over rows of the left matrix
    for (int i = 0; i < _rows; i++)
    {
        // Iterate over columns of the right matrix
        for (int j = 0; j < other._cols; j++)
        {
            float curr_element = 0;
            for(int k = 0; k < _cols; k++)
            {
                // Accessing 1D array as 2D:
                float new_mat_cumulative_sum =
                    _data[i * _cols + k] * other._data[k * other._cols + j]
;
                curr_element += new_mat_cumulative_sum;
            }
            new_mat._data[i * other._cols + j] = curr_element;
        }
    }
    return new_mat;
}

Matrix Matrix::operator*(float scalar) const
{
    Matrix new_mat(_rows,_cols);
    for (int i = 0; i < _rows * _cols; i++)
    {
        new_mat._data[i] = _data[i] * scalar;
    }
    return  new_mat;
}

Matrix operator*(float scalar, const Matrix &matrix)
{
    // Multiplying a matrix by scalar is a commutative action
    // therefore we can use previous function.
    return matrix * scalar;
}

float Matrix::operator()(int i, int j) const
{
    if (i >= _rows|| j >= _cols|| i < 0 || j < 0)
    {
        throw std::out_of_range("Invalid indices");
    }
    return _data[i * _cols + j];
}

float & Matrix::operator()(int i, int j)
{
    if (i >= _rows|| j >= _cols|| i < 0 || j < 0)
    {
        throw std::out_of_range("Invalid indices");
    }
    return _data[i * _cols + j];
}

float Matrix::operator[](int i) const
{
    if (i >= _rows * _cols|| i < 0 )
    {
        throw std::out_of_range("Invalid indices");
    }
    return _data[i];
}

float & Matrix::operator[](int i)
{
    if (i >= _rows * _cols|| i < 0 )
    {
        throw std::out_of_range("Invalid indices");
    }
    return _data[i];
}

std::ostream & operator<<(std::ostream &os, const Matrix &m)
{
    for (int i = 0; i < m._rows; i++)
    {
        for (int j = 0; j < m._cols; j++)
        {
            if (m._data[i * m._cols + j] > 0.1)
            {
                os << "**";
            }
            else
            {
                os << "  ";
            }

        }
        os << std::endl;
    }
    return os;
}

std::istream & operator>>(std::istream &is, Matrix &m)
{
    // Casting explicitly to char* as shown in tutorialspoint.
    long int total_bytes = m._rows * m._cols * sizeof(float);
    is.read(reinterpret_cast<char*>(m._data), total_bytes);
    return is;
}

int Matrix::get_rows() const
{
    return _rows;
}

int Matrix::get_cols() const
{
    return _cols;
}

void Matrix::plain_print() const
{
    for (int i = 0; i < _rows ; i++)
    {
        for ( int j = 0; j < _cols ; j++)
        {
            std::cout << _data[i * _cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix & Matrix::transpose()
{
    float* new_data_arr = new float[_rows * _cols];
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            // swapping row and col of each float from original array
            new_data_arr[j * _rows + i] = _data[i * _cols + j];
        }
    }
    delete[] _data;
    _data = new_data_arr;
    std::swap(_rows, _cols);
    return *this;
}

Matrix & Matrix::vectorize()
{
    _rows = _rows * _cols;
    _cols = 1;
    return  *this;
}

Matrix Matrix::dot(const Matrix &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::length_error("Dimensions of matrices don't match");
    }
    Matrix new_mat(_rows, _cols);
    for (int i = 0; i < _rows * _cols; i++)
    {
        new_mat._data[i] = _data[i] * other._data[i];
    }
    return new_mat;
}

float Matrix::norm() const
{
    float total_sum = 0;
    for (int i = 0; i < _rows * _cols; i++)
    {
        total_sum += _data[i] * _data[i];
    }
    return std::sqrt(total_sum);
}

int Matrix::argmax() const
{
    int max_index = 0;
    float max_val = _data[0];
    for (int i = 1; i < _rows * _cols; i++)
    {
        if (_data[i] > max_val)
        {
            max_val = _data[i];
            max_index = i;
        }
    }
    return max_index;
}

float Matrix::sum() const
{
    float sum = 0;
    for (int i = 0; i < _rows * _cols; i++)
    {
        sum += _data[i];
    }
    return sum;
}

void Matrix::swap_matrix_rows(int row1, int row2)
{
    for (int j = 0; j < _cols; j++)
    {
        std::swap(_data[row1 * _cols + j], _data[row2 * _cols + j]);
    }
}

void Matrix::divide_row(int row_num, float scalar)
{
    for (int j = 0; j < _cols; j++)
    {
        _data[row_num * _cols + j] /= scalar;
    }
}

void Matrix::subtract_row(int target_row, int src_row, float scalar, int col_start)
{
    for ( int j = col_start; j < _cols; j++)
    {
        _data[target_row * _cols + j] -= (_data[src_row * _cols + j] * scalar);
    }
}

int Matrix::find_pivot_row(int curr_col, int start_row) const
{
    int max_row_index = start_row;
    float max_val = std::abs(_data[start_row * _cols + curr_col]);
    for (int i = start_row + 1; i < _rows; i++)
    {
        float curr_val = std::abs(_data[i * _cols + curr_col]);
        if (curr_val > max_val)
        {
            max_row_index = i;
            max_val = curr_val;
        }
    }
    return max_row_index;
}

Matrix Matrix::rref() const
{
    Matrix new_mat = *this;
    int curr_col = 0; // The column we are currently trying to clear
    int curr_row = 0; // The row where we want to place our next pivot
    while (curr_row < _rows && curr_col < _cols)
    {
        int pivot_row = new_mat.find_pivot_row(curr_col, curr_row);
        // If the entire column is 0 we cannot place a pivot here,
        // so we move to next column.
        if (std::abs(new_mat._data[pivot_row * _cols + curr_col]) == 0)
        {
            curr_col++;
            continue;
        }
        // swap the pivot row with current row.
        if (pivot_row != curr_row)
        {
            new_mat.swap_matrix_rows(curr_row, pivot_row);
        }
        // Normalize the pivot row by dividing by pivot_value.
        float pivot_value = new_mat._data[curr_row * _cols + curr_col];
        new_mat.divide_row(curr_row, pivot_value);

        // turn all other entries in this column to 0.
        for (int i = 0; i < _rows; i++)
        {
            if (i != curr_row)
            {
                float scalar = new_mat._data[i * _cols + curr_col];
                new_mat.subtract_row(i, curr_row, scalar, curr_col);
            }
        }
        curr_row++;
        curr_col++;
    }
    return new_mat;
}
