// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * @brief Matrix struct for matrix dimensions.
 */
struct matrix_dims
{
    int rows;
    int cols;
};

/**
* @brief Matrix class for neural network implementation.
* Represents a matrix of float values and supports various linear algebra operations.
*/
class Matrix
{
    int _rows;
    int _cols;
    float* _data;


    // Private helper functions.
    /**
    * @brief Swaps two rows in the matrix.
     * Used in rref() to bring the pivot to the correct position.
     * @param row1 Index of the first row.
     * @param row2 Index of the second row.
     */
    void swap_matrix_rows(int row1, int row2);

    /**
     * @brief Divides all elements in a specific row by a scalar.
     * Used to normalize the pivot row making the leading element 1.
     * @param row_num The index of the row to divide.
     * @param scalar The value to divide by (assumed non-zero).
     */
    void divide_row(int row_num, float scalar);

    /**
     * @brief Subtracts a scaled source row from a target row.
     * Performs the elementary row operation: R_target = R_target - (scalar * R_src).
     * @param target_row The row to be changed.
     * @param src_row The pivot row.
     * @param scalar The factor to multiply the source row by.
     * @param col_start Optimization: Start the operation from this column index
     * (since previous columns are known to be zero).
     */
    void subtract_row(int target_row, int src_row, float scalar, int col_start);

    /**
     * @brief Finds the row with the largest absolute value in a specific column.
     * Searches from start_row down to the last row.
     * @param curr_col The column to search in.
     * @param start_row The row index to start searching from.
     * @return The index of the row containing the pivot.
     * @note Used in order to choose the largest available scalar to divide
     * The pivot row by, in order to improve stability (dividing by very small
     * Scalar will potentially result in very large floats).
     */
    int find_pivot_row(int curr_col, int start_row) const;

public:
    // Constructors

    /**
     * @brief Constructs a Matrix of size rows x cols. Inits all elements to 0.
     * @param rows Number of rows (must be > 0).
     * @param cols Number of columns (must be > 0).
     * @throw std::length_error if rows or cols are non-positive.
     */
    Matrix(int rows, int cols);

    /**
     * @brief Default constructor. Constructs a Matrix of size 1x1.
     * Initializes the single element to 0.
     */
    Matrix();

    /**
     * @brief Copy constructor.
     * Constructs a matrix from another Matrix "other".
     * @param other The matrix to copy from.
     */
    Matrix(const Matrix& other);

    /**
     * @brief Destructor. Releases all resources owned by the class.
     */
    ~Matrix();

    // Operators
    /**
     * @brief Assignment operator. Assigns "other" to "this".
     * Handles memory management and self-assignment.
     * @param other The matrix to assign from.
     * @return Reference to this matrix.
     */
    Matrix& operator=(const Matrix& other);

    /**
     * @brief Matrix addition.
     * Calculates the addition of "this" and "other" and assigns it to "this".
     * @param other The matrix to add.
     * @return Reference to this matrix, supports chaining.
     * @throw std::length_error if dimensions do not match.
     */
    Matrix& operator+=(const Matrix& other);

    /**
     * @brief Matrix addition.
     * Calculates addition of "this" and "other".
     * @param other The matrix to add.
     * @return A new Matrix representing the sum.
     * @throw std::length_error if dimensions do not match.
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Matrix multiplication.
     * Calculates Matrix Multiplication (A * B).
     * @param other The matrix to multiply with (rhs).
     * @return A new Matrix representing the product.
     * @throw std::length_error if this->cols != other.rows.
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Scalar multiplication on the right (M * c).
     * @param scalar The float scalar to multiply by.
     * @return A new Matrix where each element is the product of the scalar
     * and the matrix element.
     */
    Matrix operator*(float scalar) const; //scalar multiplication on the right

    /**
     * @brief Scalar multiplication on the left (c * M).
     * @param scalar The float scalar to multiply by.
     * @param matrix The matrix to be multiplied.
     * @return A new Matrix where each element is the product of the scalar
     * and the matrix element.
     */
    friend Matrix operator*(float scalar,const Matrix& matrix);

    // Index operators

    /**
     * @brief Parenthesis indexing (read-only).
     * Access element at (i, j).
     * @param i Row index.
     * @param j Column index.
     * @return The value at (i, j).
     * @throw std::out_of_range if indices are invalid.
     */
    float operator()(int i, int j) const;

    /**
     * @brief Parenthesis indexing (writing access).
     * Access element at (i, j).
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the value at (i, j).
     * @throw std::out_of_range if indices are invalid.
     */
    float& operator()(int i, int j);

    /**
     * @brief Brackets indexing (read-only).
     * Access the k-th element as if the matrix is a vector.
     * @param i The index in the flattened array.
     * @return The value at index i.
     * @throw std::out_of_range if index is invalid.
     */
    float operator[](int i) const;

    /**
     * @brief Brackets indexing (writing access).
     * Access the k-th element as if the matrix is a vector.
     * @param i The index in the flattened array.
     * @return Reference to the value at index i.
     * @throw std::out_of_range if index is invalid.
     */
    float& operator[](int i);

    // Input / Output operators

    /**
     * @brief Output stream operator.
     * Prints the matrix (double asterisks for values > 0.1, spaces otherwise).
     * @param os The output stream.
     * @param m The matrix to print.
     * @return Reference to the output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    /**
     * @brief Input stream operator.
     * Fills the matrix elements from an input stream.
     * @param is The input stream.
     * @param m The matrix to fill.
     * @return Reference to the input stream.
     * @note Does not assume the stream contains enough data.
     */
    friend std::istream& operator>>(std::istream& is, Matrix& m);

    // Getters and plain print

    /**
     * @brief Gets the number of rows.
     * @return Number of rows.
     */
    int get_rows() const;

    /**
     * @brief Gets the number of columns.
     * @return Number of columns.
     */
    int get_cols() const;

    /**
     * @brief Prints all matrix elements to stdout.
     * Prints space after each element and newline after each row.
     */
    void plain_print() const;

    //mathematical methods

    /**
     * @brief Transposes this matrix in place.
     * @return Reference to this matrix (supports chaining).
     */
    Matrix& transpose();

    /**
     * @brief Vectorizes this matrix in place.
     * Transforms the matrix into a column vector with (rows * cols) rows.
     * @return Reference to this matrix (supports chaining).
     */
    Matrix& vectorize();

    /**
     * @brief Returns the element-wise multiplication.
     * @param other The matrix to multiply with.
     * @return A new Matrix resulting from the dot product.
     * @throw std::length_error if dimensions do not match.
     */
    Matrix dot(const Matrix& other) const;

    /**
     * @brief Calculates the Frobenius norm of the matrix.
     * @return The norm value.
     */
    float norm() const;

    /**
     * @brief Returns the index of the largest number in the matrix.
     * If there's a tie, returns the lower index.
     * @return The index (0 to size-1).
     */
    int argmax() const;

    /**
     * @brief Calculates the sum of all elements in the matrix.
     * @return The sum.
     */
    float sum() const;

    /**
     * @brief Calculates the Reduced Row Echelon Form.
     * @return A new Matrix that is the RREF of the original.
     * @note The original matrix remains unchanged.
     */
    Matrix rref() const;
};


#endif //MATRIX_H