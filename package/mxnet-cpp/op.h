/*!
*  Copyright (c) 2016 by Contributors
* \file op.h
* \brief definition of all the operators
* \author Chuntao Hong, Xin Li
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/op_util.h"
#include "mxnet-cpp/operator.h"
#include "dmlc/optional.h"

namespace mxnet {
namespace cpp {

/*!
 * \breif Applies the softmax function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *
 *        .. math::
 *        softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        Example::
 *
 *        x = [[ 1.  1.  1.]
 *        [ 1.  1.  1.]]
 *
 *        softmax(x,axis=0) = [[ 0.5  0.5  0.5]
 *        [ 0.5  0.5  0.5]]
 *
 *        softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
 *        [ 0.33333334,  0.33333334,  0.33333334]]
 *
 *
 *
 *        Defined in src/operator/nn/softmax.cc:L35
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \return new symbol
 */
inline Symbol softmax(const std::string& symbol_name,
                      Symbol data,
                      int axis = -1) {
  return Operator("softmax")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the log softmax of the input.
 *        This is equivalent to computing softmax followed by log.
 *
 *        Examples::
 *
 *        >>> x = mx.nd.array([1, 2, .1])
 *        >>> mx.nd.log_softmax(x).asnumpy()
 *        array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
 *
 *        >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
 *        >>> mx.nd.log_softmax(x, axis=0).asnumpy()
 *        array([[-0.34115392, -0.69314718, -1.24115396],
 *        [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
 *
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \return new symbol
 */
inline Symbol log_softmax(const std::string& symbol_name,
                          Symbol data,
                          int axis = -1) {
  return Operator("log_softmax")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L32
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L104
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L137
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reshapes the input array.
 *
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4).The size of the new shape
 *
 *        Example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *
 *        - ``0``  copy this dimension from the input to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *
 *        Example::
 *
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L87
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(const std::string& symbol_name,
                      Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *
 *        Example::
 *
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L127
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Permutes the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L168
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(const std::string& symbol_name,
                        Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inserts a new axis of size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L204
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Position where new axis is to be inserted. Suppose that the input
 *        `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim,
 * \return new symbol
 */
inline Symbol expand_dims(const std::string& symbol_name,
                          Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slices a contiguous region of the array.
 *
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *
 *        This function returns a sliced continuous region of the array between the
 *        by `begin` and `end`.
 *
 *        For an input array of `n` dimensions, slice operation with ``begin=(b_0,
 *        and ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape
 *        ``(e_1-b_0, ..., e_n-b_n-1)``.
 *
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        Example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L244
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \return new symbol
 */
inline Symbol slice(const std::string& symbol_name,
                    Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slices along a given axis.
 *
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *
 *        Examples::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L324
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(const std::string& symbol_name,
                         Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *        Example::
 *
 *        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
 *        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
 *        dot(x,y)[0,0,1,1] = 0
 *        sum(x[0,0,:]*y[:,1,1]) = 0
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L363
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L399
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Clips (limits) the values in an array.
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_x` would be::
 *
 *        clip(x, a_min, a_max) = max(min(x, a_max), a_min))
 *
 *        Example::
 *
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L444
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(const std::string& symbol_name,
                   Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeats elements of an array.
 *
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L486
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(const std::string& symbol_name,
                     Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeats the whole array multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L547
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(const std::string& symbol_name,
                   Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the order of elements along given axis while preserving array shape.
 *
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L588
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(const std::string& symbol_name,
                      Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return an array of zeros with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        zeros_like(x) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol zeros_like(const std::string& symbol_name,
                         Symbol data) {
  return Operator("zeros_like")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return an array of ones with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        ones_like(x) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol ones_like(const std::string& symbol_name,
                        Symbol data) {
  return Operator("ones_like")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Adds all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *
 *        Defined in src/operator/tensor/elemwise_sum.cc:L63
 * \param symbol_name name of the resulting symbol
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::string& symbol_name,
                    const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns indices of the maximum values along an axis.
 *
 *        In the case of multiple occurrences of maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmax along axis 0
 *        argmax(x, axis=0) = [ 1.,  1.,  1.]
 *
 *        // argmax along axis 1
 *        argmax(x, axis=1) = [ 2.,  2.]
 *
 *        // argmax along axis 1 keeping same dims as an input array
 *        argmax(x, axis=1, keepdims=True) = [[ 2.],
 *        [ 2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L33
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmax(const std::string& symbol_name,
                     Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns indices of the minimum values along an axis.
 *
 *        In the case of multiple occurrences of minimum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmin along axis 0
 *        argmin(x, axis=0) = [ 0.,  0.,  0.]
 *
 *        // argmin along axis 1
 *        argmin(x, axis=1) = [ 0.,  0.]
 *
 *        // argmin along axis 1 keeping same dims as an input array
 *        argmin(x, axis=1, keepdims=True) = [[ 0.],
 *        [ 0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L58
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmin(const std::string& symbol_name,
                     Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns argmax indices of each channel from the input array.
 *
 *        The result will be an NDArray of shape (num_channel,).
 *
 *        In case of multiple occurrences of the maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        argmax_channel(x) = [ 2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L78
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \return new symbol
 */
inline Symbol argmax_channel(const std::string& symbol_name,
                             Symbol data) {
  return Operator("argmax_channel")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Picks elements from an input array according to the input indices along the
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last element along an axis (the `clip` mode).
 *
 *        This function supports n-dimensional input and (n-1)-dimensional indices arrays.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // picks elements with specified indices along axis 0
 *        pick(x, y=[0,1], 0) = [ 1.,  4.]
 *
 *        // picks elements with specified indices along axis 1
 *        pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
 *
 *        y = [[ 1.],
 *        [ 0.],
 *        [ 2.]]
 *
 *        // picks elements with specified indices along axis 1 and dims are maintained
 *        pick(x,y, 1, keepdims=True) = [[ 2.],
 *        [ 3.],
 *        [ 6.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L126
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param index The index array
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol pick(const std::string& symbol_name,
                   Symbol data,
                   Symbol index,
                   dmlc::optional<int> axis = dmlc::optional<int>(),
                   bool keepdims = false) {
  return Operator("pick")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .SetInput("index", index)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns result of first array elements raised to powers from second array,
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_power(x, y) = [[ 2.,  2.,  2.],
 *        [ 4.,  4.,  4.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L26
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_power(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise maximum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L61
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_maximum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise minimum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L96
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_minimum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hypotenuse of a right angled triangle, given its "legs"
 *        with broadcasting.
 *
 *        It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
 *
 *        Example::
 *
 *        x = [[ 3.,  3.,  3.]]
 *
 *        y = [[ 4.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
 *        [ 5.,  5.,  5.]]
 *
 *        z = [[ 0.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
 *        [ 5.,  5.,  5.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L137
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_hypot(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the sum of array elements over given axes.
 *
 *        .. Note::
 *
 *        `sum` and `sum_axis` are equivalent.
 *
 *        Example::
 *
 *        data = [[[1,2],[2,3],[1,3]],
 *        [[1,4],[4,3],[5,2]],
 *        [[7,1],[7,2],[7,3]]]
 *
 *        sum(data, axis=1)
 *        [[  4.   8.]
 *        [ 10.   9.]
 *        [ 21.   6.]]
 *
 *        sum(data, axis=[1,2])
 *        [ 12.  19.  27.]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L51
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol sum(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the mean of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L64
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol mean(const std::string& symbol_name,
                   Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the product of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol prod(const std::string& symbol_name,
                   Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the sum of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L92
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nansum(const std::string& symbol_name,
                     Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false,
                     bool exclude = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the product of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L107
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nanprod(const std::string& symbol_name,
                      Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false,
                      bool exclude = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the max of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L121
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol max(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the min of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L135
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol min(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcasts the input array over particular axes.
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        Example::
 *
 *        // given x of shape (1,2,1)
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast x on on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast x on on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L168
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(const std::string& symbol_name,
                             Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcasts the input array to a new shape.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimension which you do not want to change can also be kept as `0` which
 *        So with `shape=(2,0)`, we will obtain the same result as in the above example.
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L192
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(const std::string& symbol_name,
                           Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flattens the input array and then computes the l2 norm.
 *
 *        Examples::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        norm(x) = [5.47722578]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L218
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \return new symbol
 */
inline Symbol norm(const std::string& symbol_name,
                   Symbol data) {
  return Operator("norm")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 */
enum class TopkRetTyp {
  kBoth = 0,
  kIndices = 1,
  kMask = 2,
  kValue = 3
};

/*!
 * \breif Returns the top *k* elements in an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // returns an index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // returns the value of top-2 largest elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // returns the value of top-2 smallest elements on last axis
 *        topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
 *        [ 0.1 ,  0.2]]
 *
 *        // returns the value of top-2 largest elements on axis 0
 *        topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
 *        [ 0.1,  0.2,  0.2]]
 *
 *        // flattens and then returns list of both values and indices
 *        topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [
 *
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L44
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 * \param is_ascend Whether to choose k largest or k smallest elements. Top K largest
 * \return new symbol
 */
inline Symbol topk(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::kIndices,
                   bool is_ascend = false) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns a sorted copy of an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sorts along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flattens and then sorts
 *        sort(x) = [ 1.,  1.,  3.,  4.]
 *
 *        // sorts along the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L107
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the indices that would sort an input array along the given axis.
 *
 *        This function performs sorting along the given axis and returns an array of
 *        as an input array that index data in sorted order.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L157
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol argsort(const std::string& symbol_name,
                      Symbol data,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true) {
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Adds arguments element-wise.
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes rectified linear.
 *
 *        .. math::
 *        max(features, 0)
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L18
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(const std::string& symbol_name,
                   Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(const std::string& symbol_name,
                      Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L91
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(const std::string& symbol_name,
                        Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Stops gradient computation.
 *        .. note:: ``make_loss`` is deprecated, use ``MakeLoss``.
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L98
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(const std::string& symbol_name,
                        Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Output data type.
 */
enum class CastDtype {
  kFloat16 = 0,
  kFloat32 = 1,
  kFloat64 = 2,
  kInt32 = 3,
  kUint8 = 4
};

/*!
 * \breif Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L155
 * \param symbol_name name of the resulting symbol
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(const std::string& symbol_name,
                   Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Negate src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:174
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(const std::string& symbol_name,
                       Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L186
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L201
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L216
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L232
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(const std::string& symbol_name,
                   Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise ceiling of the input.
 *
 *        The ceil of the scalar x is the smallest integer i, such that i >= x.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L245
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise floor of the input.
 *
 *        The floor of the scalar x is the largest integer i, such that i <= x.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L258
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return the element-wise truncated value of the input.
 *
 *        The truncated value of the scalar x is the nearest integer i which is closer to
 *        zero than x is. In short, the fractional part of the signed number x is
 *
 *        Example::
 *
 *        trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L272
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol trunc(const std::string& symbol_name,
                    Symbol data) {
  return Operator("trunc")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer towards zero of the
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L283
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(const std::string& symbol_name,
                  Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [3, 9, 16]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L297
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L315
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L333
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [inf, 1, 0.707]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L352
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L362
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L372
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L382
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(const std::string& symbol_name,
                   Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L398
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L412
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L425
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(const std::string& symbol_name,
                    Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L441
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L457
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(const std::string& symbol_name,
                  Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L474
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L491
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L507
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L521
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(const std::string& symbol_name,
                      Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L535
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(const std::string& symbol_name,
                      Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L549
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L563
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L577
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic sine of the input array, computed
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L587
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic cosine of the input array, computed
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L597
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic tangent of the input array,
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L607
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the gamma function (extension of the factorial function to the reals) ,
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:617
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(const std::string& symbol_name,
                    Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise log of the absolute value of the gamma function of the
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:627
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(const std::string& symbol_name,
                      Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Data type of weight.
 */
enum class EmbeddingDtype {
  kFloat16 = 0,
  kFloat32 = 1,
  kFloat64 = 2,
  kInt32 = 3,
  kUint8 = 4
};

/*!
 * \breif Maps integer indices to vector representations (embeddings).
 *
 *        This operator maps words to real-valued vectors in a high-dimensional space,
 *        called word embeddings. These embeddings can capture semantic and syntactic
 *        For example, it has been noted that in the learned embedding spaces, similar
 *        to be close to each other and dissimilar words far apart.
 *
 *        For an input array of shape (d1, ..., dK),
 *        the shape of an output array is (d1, ..., dK, output_dim).
 *        All the input values should be integers in the range [0, input_dim).
 *
 *        If the input_dim is ip0 and output_dim is op0, then shape of the embedding
 *        (ip0, op0).
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last vector in an embedding matrix.
 *
 *        Examples::
 *
 *        input_dim = 4
 *        output_dim = 5
 *
 *        // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
 *        y = [[  0.,   1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.,   9.],
 *        [ 10.,  11.,  12.,  13.,  14.],
 *        [ 15.,  16.,  17.,  18.,  19.]]
 *
 *        // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
 *        x = [[ 1.,  3.],
 *        [ 0.,  2.]]
 *
 *        // Mapped input x to its vector representation y.
 *        Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
 *        [ 15.,  16.,  17.,  18.,  19.]],
 *
 *        [[  0.,   1.,   2.,   3.,   4.],
 *        [ 10.,  11.,  12.,  13.,  14.]]]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L55
 * \param symbol_name name of the resulting symbol
 * \param data The input array to the embedding operator.
 * \param weight The embedding weight matrix.
 * \param input_dim Vocabulary size of the input indices.
 * \param output_dim Dimension of the embedding vectors.
 * \param dtype Data type of weight.
 * \return new symbol
 */
inline Symbol Embedding(const std::string& symbol_name,
                        Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim,
                        EmbeddingDtype dtype = EmbeddingDtype::kFloat32) {
  static const char *EmbeddingDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetParam("dtype", EmbeddingDtypeValues[int(dtype)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol(symbol_name);
}

/*! \breif Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if
 *        all indices mentioned are too large, they are replaced by the index that
 *        addresses the last element along an axis.  "wrap" means to wrap around.
 */
enum class TakeMode {
  kClip = 0,
  kRaise = 1,
  kWrap = 2
};

/*!
 * \breif Takes elements from an input array along the given axis.
 *
 *        This function slices the input array along a particular axis with the provided
 *
 *        Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0,
 *        will have shape ``(i0, i1, d1, d2)``, computed by::
 *
 *        output[i,j,:,:] = input[indices[i,j],:,:]
 *
 *        .. note::
 *        - `axis`- Only slicing along axis 0 is supported for now.
 *        - `mode`- Only `clip` mode is supported for now.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices along axis 0
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L117
 * \param symbol_name name of the resulting symbol
 * \param a The input array.
 * \param indices The indices of the values to be extracted.
 * \param axis The axis of input array to be taken.
 * \param mode Specify how out-of-bound indices bahave. "clip" means clip to the range.
 *        So, if all indices mentioned are too large, they are replaced by the index that
 *        addresses the last element along an axis.  "wrap" means to wrap around.
 * \return new symbol
 */
inline Symbol take(const std::string& symbol_name,
                   Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::kClip) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Takes elements from a data batch.
 *
 *        .. note::
 *        `batch_take` is deprecated. Use `pick` instead.
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L172
 * \param symbol_name name of the resulting symbol
 * \param a The input array
 * \param indices The index array
 * \return new symbol
 */
inline Symbol batch_take(const std::string& symbol_name,
                         Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output
 */
enum class One_hotDtype {
  kFloat16 = 0,
  kFloat32 = 1,
  kFloat64 = 2,
  kInt32 = 3,
  kUint8 = 4
};

/*!
 * \breif Returns a one-hot array.
 *
 *        The locations represented by `indices` take value `on_value`, while all
 *        other locations take value `off_value`.
 *
 *        `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d``
 *        in an output array of shape ``(i0, i1, d)`` with::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L218
 * \param symbol_name name of the resulting symbol
 * \param indices array of locations where to set on_value
 * \param depth Depth of the one hot dimension.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(const std::string& symbol_name,
                      Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::kFloat32) {
  static const char *One_hotDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **equal to** (==) comparison operation with
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L27
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_equal(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **not equal to** (!=) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L45
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_not_equal(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **greater than** (>) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L63
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **greater than or equal to** (>=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L81
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(const std::string& symbol_name,
                                      Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **lesser than** (<) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L99
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser(const std::string& symbol_name,
                               Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the result of element-wise **lesser than or equal to** (<=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L117
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(const std::string& symbol_name,
                                     Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs general matrix multiplication and accumulation.
 *        Input are three tensors *A*, *B*, *C* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\ , *C*\ :sub:`i` be the matrices given by the
 *        The operator performs the BLAS3 function *gemm*
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ )
 *
 *        on all such triples of matrices. Here *alpha* and *beta* are scalar operator
 *        is either the identity or the matrix transposition.
 *
 *        In case of *n=2*, a single *gemm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply-add
 *        A = [[1.0, 1.0], [1.0, 1.0]]
 *        B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
 *        C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *        linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
 *        = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]
 *
 *        // Batch matrix multiply-add
 *        A = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        B = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        C = [[[10.0]], [[0.01]]]
 *        linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
 *        = [[[104.0]], [[0.14]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L48
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of input matrices
 * \param B Tensor of input matrices
 * \param C Tensor of input matrices
 * \param transpose_a Multiply with transposed of first input (A).
 * \param transpose_b Multiply with transposed of second input (B).
 * \param alpha Scalar factor multiplied with A*B.
 * \param beta Scalar factor multiplied with C.
 * \return new symbol
 */
inline Symbol linalg_gemm(const std::string& symbol_name,
                          Symbol A,
                          Symbol B,
                          Symbol C,
                          bool transpose_a = false,
                          bool transpose_b = false,
                          double alpha = 1,
                          double beta = 1) {
  return Operator("linalg_gemm")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetInput("A", A)
           .SetInput("B", B)
           .SetInput("C", C)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs general matrix multiplication.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *gemm* (restricted to two arguments)
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ )
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter and
 *        the identity or the matrix transposition.
 *
 *        In case of *n=2*, a single *gemm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply
 *        A = [[1.0, 1.0], [1.0, 1.0]]
 *        B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
 *        linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0)
 *        = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
 *
 *        // Batch matrix multiply
 *        A = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        B = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0 )
 *        = [[[4.0]], [[0.04 ]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L106
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of input matrices
 * \param B Tensor of input matrices
 * \param transpose_a Multiply with transposed of first input (A).
 * \param transpose_b Multiply with transposed of second input (B).
 * \param alpha Scalar factor multiplied with A*B.
 * \return new symbol
 */
inline Symbol linalg_gemm2(const std::string& symbol_name,
                           Symbol A,
                           Symbol B,
                           bool transpose_a = false,
                           bool transpose_b = false,
                           double alpha = 1) {
  return Operator("linalg_gemm2")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs Cholesky factorization of a symmetric positive-definite matrix.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator performs the Cholesky factorization (LAPACK function *potrf*)
 *        on each *A*\ :sub:`i`\ ,
 *        i.e. it computes a lower triangular matrix *U*\ :sub:`i` such that
 *
 *        *A*\ :sub:`i`\  = *U*\ :sub:`i`\  \* *U*\ :sub:`i`\ \ :sup:`T`
 *
 *        for all such matrices. The matrices *A*\ :sub:`i` must be all symmetric and
 *        The resulting matrices *U*\ :sub:`i` will contain zeros in the upper triangle
 *        apart from the diagonal.
 *
 *        In case of *n=2*, a single Cholesky factorization is performed on the matrix
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix factorization
 *        A = [[4.0, 1.0], [1.0, 4.25]]
 *        linalg_potrf(A) = [[2.0, 0], [0.5, 2.0]]
 *
 *        // Batch matrix factorization
 *        A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
 *        linalg_potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L159
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of input matrices to be decomposed
 * \return new symbol
 */
inline Symbol linalg_potrf(const std::string& symbol_name,
                           Symbol A) {
  return Operator("linalg_potrf")
           .SetInput("A", A)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs matrix inversion from a Cholesky factorization.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator assumes that each *A*\ :sub:`i` is the Cholesky factorization of
 *        positive-definite matrix *B*\ :sub:`i` given as a lower triangular matrix
 *        (so *A* is the output of a prior call to operator *linalg_potrf*). The operator
 *        inverse of each *B*\ :sub:`i` from this decomposition, i.e
 *
 *        *out*\ :sub:`i` = *B*\ :sub:`i`\ \ :sup:`-1`
 *
 *        for all such matrices.
 *
 *        In case of *n=2*, the operation is performed on the matrix *A* itself.
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix inverse
 *        A = [[2.0, 0], [0.5, 2.0]]
 *        linalg_potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]
 *
 *        // Batch matrix inverse
 *        A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
 *        linalg_potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
 *        [[0.06641, -0.01562], [-0.01562, 0,0625]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L211
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of lower triangular matrices
 * \return new symbol
 */
inline Symbol linalg_potri(const std::string& symbol_name,
                           Symbol A) {
  return Operator("linalg_potri")
           .SetInput("A", A)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs multiplication with a triangular matrix.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *trmm*
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *B*\ :sub:`i`
 *
 *        or
 *
 *        *out*\ :sub:`i` = *alpha* \* *B*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ )
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter,
 *        the identity or the matrix transposition (depending on the parameter
 *        order of matrix multiplication depends on the parameter *rightside*.
 *        All matrices *A*\ :sub:`i` must be lower triangular.
 *
 *        In case of *n=2*, a single *trmm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply
 *        A = [[1.0, 0], [1.0, 1.0]]
 *        B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *        linalg_trmm(A, B, alpha = 2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
 *
 *        // Batch matrix multiply
 *        A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
 *        B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
 *        linalg_trmm(A, B, alpha = 2.0 ) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
 *        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
 *
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L268
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of lower triangular matrices
 * \param B Tensor of matrices
 * \param transpose Use transposed of the triangular matrix
 * \param rightside Multiply triangular matrix from the right to non-triangular one.
 * \param alpha Scalar factor to be applied to the result.
 * \return new symbol
 */
inline Symbol linalg_trmm(const std::string& symbol_name,
                          Symbol A,
                          Symbol B,
                          bool transpose = false,
                          bool rightside = false,
                          double alpha = 1) {
  return Operator("linalg_trmm")
           .SetParam("transpose", transpose)
           .SetParam("rightside", rightside)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Solves matrix equations involving a triangular matrix.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *trsm*, i.e. it solves the equation
 *
 *        *op*\ (*A*\ :sub:`i`\ ) \* *X*\ :sub:`i` = *alpha* \* *B*\ :sub:`i`
 *
 *        or
 *
 *        *X*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ ) = *alpha* \* *B*\ :sub:`i`
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter,
 *        the identity or the matrix transposition (depending on the parameter
 *        order of multiplication on the left depends on the parameter *rightside*.
 *        All matrices *A*\ :sub:`i` must be lower triangular.
 *
 *        In case of *n=2*, a single *trsm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix solve
 *        A = [[1.0, 0], [1.0, 1.0]]
 *        B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
 *        linalg_trsm(A, B, alpha = 0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *
 *        // Batch matrix solve
 *        A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
 *        B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
 *        [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
 *        linalg_trsm(A, B, alpha = 0.5 ) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
 *        [[2.0, 2.0, 2.0 ], [2.0, 2.0, 2.0]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L331
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of lower triangular matrices
 * \param B Tensor of matrices
 * \param transpose Use transposed of the triangular matrix
 * \param rightside Multiply triangular matrix from the right to non-triangular one.
 * \param alpha Scalar factor to be applied to the result.
 * \return new symbol
 */
inline Symbol linalg_trsm(const std::string& symbol_name,
                          Symbol A,
                          Symbol B,
                          bool transpose = false,
                          bool rightside = false,
                          double alpha = 1) {
  return Operator("linalg_trsm")
           .SetParam("transpose", transpose)
           .SetParam("rightside", rightside)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the sum of the logarithms of all diagonal elements in a matrix.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator performs a reduction of each such matrix to a scalar by summing up
 *        of all diagonal elements. All matrices must be square and all diagonal elements
 *
 *        In case of *n=2*, *A* represents a single matrix on which the reduction will be
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix reduction
 *        A = [[1.0, 1.0], [1.0, 7.0]]
 *        linalg_sumlogdiag(A) = [1.9459]
 *
 *        // Batch matrix reduction
 *        A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
 *        linalg_sumlogdiag(A) = [1.9459, 3.9318]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L379
 * \param symbol_name name of the resulting symbol
 * \param A Tensor of square matrices
 * \return new symbol
 */
inline Symbol linalg_sumlogdiag(const std::string& symbol_name,
                                Symbol A) {
  return Operator("linalg_sumlogdiag")
           .SetInput("A", A)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        from x or y, depending on the elements from condition are true or false. x and
 *        y must have the same shape. If condition has the same shape as x, each element
 *        in the output array is from x if the corresponding element in the condition is
 *        true, and from y if false. If condition does not have the same shape as x, it
 *        must be a 1D array whose size is the same as x's first dimension size. Each row
 *        of the output array is from x's row if the corresponding element from condition
 *
 *        From:src/operator/tensor/control_flow_op.cc:21
 * \param symbol_name name of the resulting symbol
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(const std::string& symbol_name,
                    Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar) by summing
 *
 *        .. math::
 *
 *        f(x) =
 *        \begin{cases}
 *        (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
 *        |x|-0.5/\sigma^2,& \text{otherwise}
 *        \end{cases}
 *
 *        where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the
 *
 *        Example::
 *
 *        smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L79
 * \param symbol_name name of the resulting symbol
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(const std::string& symbol_name,
                        Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Only support int32 for now.
 */
enum class Sample_multinomialDtype {
  kInt32 = 0
};

/*!
 * \breif Concurrent sampling from multiple multinomial distributions.
 *
 *        *data* is an *n* dimensional array whose last dimension has length *k*, where
 *        *k* is the number of possible outcomes of each multinomial distribution. This
 *        operator will draw *shape* samples from each distribution. If shape is empty
 *        one sample will be drawn from each distribution.
 *
 *        If *get_prob* is true, a second array containing log likelihood of the drawn
 *        samples will also be returned. This is usually used for reinforcement learning
 *        where you can provide reward as head gradient for this array to estimate
 *        gradient.
 *
 *        Note that the input distribution must be normalized, i.e. *data* must sum to
 *        1 along its last axis.
 *
 *        Examples::
 *
 *        probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]
 *
 *        // Draw a single sample for each distribution
 *        sample_multinomial(probs) = [3, 0]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_multinomial(probs, shape=(2)) = [[4, 2],
 *        [0, 0]]
 *
 *        // requests log likelihood
 *        sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
 *
 * \param symbol_name name of the resulting symbol
 * \param data Distribution probabilities. Must sum to one on the last axis.
 * \param shape Shape to be sampled from each random distribution.
 * \param get_prob Whether to also return the log probability of sampled result. This is
 *        usually used for differentiating through stochastic variables, e.g. in
 * \param dtype DType of the output in case this can't be inferred. Only support int32
 * \return new symbol
 */
inline Symbol sample_multinomial(const std::string& symbol_name,
                                 Symbol data,
                                 Shape shape = Shape(),
                                 bool get_prob = false,
                                 Sample_multinomialDtype dtype = Sample_multinomialDtype::kInt32) {
  static const char *Sample_multinomialDtypeValues[] = {
    "int32"
  };
  return Operator("sample_multinomial")
           .SetParam("shape", shape)
           .SetParam("get_prob", get_prob)
           .SetParam("dtype", Sample_multinomialDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_uniformDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        uniform distributions on the intervals given by *[low,high)*.
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        low = [ 0.0, 2.5 ]
 *        high = [ 1.0, 3.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_uniform(low, high) = [ 0.40451524,  3.18687344]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
 *        [ 3.18687344,  3.68352246]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L363
 * \param symbol_name name of the resulting symbol
 * \param low Lower bounds of the distributions.
 * \param high Upper bounds of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_uniform(const std::string& symbol_name,
                             Symbol low,
                             Symbol high,
                             Shape shape = Shape(),
                             Sample_uniformDtype dtype = Sample_uniformDtype::kNone) {
  static const char *Sample_uniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_uniform")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_uniformDtypeValues[int(dtype)])
           .SetInput("low", low)
           .SetInput("high", high)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_normalDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        normal distributions with parameters *mu* (mean) and *sigma* (standard
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        mu = [ 0.0, 2.5 ]
 *        sigma = [ 1.0, 3.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_normal(mu, sigma) = [-0.56410581,  0.95934606]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
 *        [ 0.95934606,  4.48287058]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L365
 * \param symbol_name name of the resulting symbol
 * \param mu Means of the distributions.
 * \param sigma Standard deviations of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_normal(const std::string& symbol_name,
                            Symbol mu,
                            Symbol sigma,
                            Shape shape = Shape(),
                            Sample_normalDtype dtype = Sample_normalDtype::kNone) {
  static const char *Sample_normalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_normal")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_normalDtypeValues[int(dtype)])
           .SetInput("mu", mu)
           .SetInput("sigma", sigma)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_gammaDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        gamma distributions with parameters *alpha* (shape) and *beta* (scale).
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        alpha = [ 0.0, 2.5 ]
 *        beta = [ 1.0, 0.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
 *        [ 2.25797319,  1.70734084]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L368
 * \param symbol_name name of the resulting symbol
 * \param alpha Alpha (shape) parameters of the distributions.
 * \param beta Beta (scale) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_gamma(const std::string& symbol_name,
                           Symbol alpha,
                           Symbol beta,
                           Shape shape = Shape(),
                           Sample_gammaDtype dtype = Sample_gammaDtype::kNone) {
  static const char *Sample_gammaDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_gamma")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_gammaDtypeValues[int(dtype)])
           .SetInput("alpha", alpha)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_exponentialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        exponential distributions with parameters lambda (rate).
 *
 *        The parameters of the distributions are provided as an input array.
 *        Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input array,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input value at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input array.
 *
 *        Examples::
 *
 *        lam = [ 1.0, 8.5 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_exponential(lam) = [ 0.51837951,  0.09994757]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
 *        [ 0.09994757,  0.50447971]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L370
 * \param symbol_name name of the resulting symbol
 * \param lam Lambda (rate) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_exponential(const std::string& symbol_name,
                                 Symbol lam,
                                 Shape shape = Shape(),
                                 Sample_exponentialDtype dtype = Sample_exponentialDtype::kNone) {
  static const char *Sample_exponentialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_exponential")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_exponentialDtypeValues[int(dtype)])
           .SetInput("lam", lam)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_poissonDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        Poisson distributions with parameters lambda (rate).
 *
 *        The parameters of the distributions are provided as an input array.
 *        Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input array,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input value at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input array.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        lam = [ 1.0, 8.5 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_poisson(lam) = [  0.,  13.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_poisson(lam, shape=(2)) = [[  0.,   4.],
 *        [ 13.,   8.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L372
 * \param symbol_name name of the resulting symbol
 * \param lam Lambda (rate) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_poisson(const std::string& symbol_name,
                             Symbol lam,
                             Shape shape = Shape(),
                             Sample_poissonDtype dtype = Sample_poissonDtype::kNone) {
  static const char *Sample_poissonDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_poisson")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_poissonDtypeValues[int(dtype)])
           .SetInput("lam", lam)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_negative_binomialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        negative binomial distributions with parameters *k* (failure limit) and *p*
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        k = [ 20, 49 ]
 *        p = [ 0.4 , 0.77 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_negative_binomial(k, p) = [ 15.,  16.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
 *        [ 16.,  12.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L375
 * \param symbol_name name of the resulting symbol
 * \param k Limits of unsuccessful experiments.
 * \param p Failure probabilities in each experiment.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_negative_binomial(const std::string& symbol_name,
                                       Symbol k,
                                       Symbol p,
                                       Shape shape = Shape(),
                                       Sample_negative_binomialDtype dtype = Sample_negative_binomialDtype::kNone) {
  static const char *Sample_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_negative_binomial")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_negative_binomialDtypeValues[int(dtype)])
           .SetInput("k", k)
           .SetInput("p", p)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Sample_generalized_negative_binomialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Concurrent sampling from multiple
 *        generalized negative binomial distributions with parameters *mu* (mean) and
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        mu = [ 2.0, 2.5 ]
 *        alpha = [ 1.0, 0.1 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
 *        [ 3.,  1.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L379
 * \param symbol_name name of the resulting symbol
 * \param mu Means of the distributions.
 * \param alpha Alpha (dispersion) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_generalized_negative_binomial(const std::string& symbol_name,
                                                   Symbol mu,
                                                   Symbol alpha,
                                                   Shape shape = Shape(),
                                                   Sample_generalized_negative_binomialDtype dtype = Sample_generalized_negative_binomialDtype::kNone) {
  static const char *Sample_generalized_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_generalized_negative_binomial")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_generalized_negative_binomialDtypeValues[int(dtype)])
           .SetInput("mu", mu)
           .SetInput("alpha", alpha)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_uniformDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a uniform distribution.
 *
 *        .. note:: The existing alias ``uniform`` is deprecated.
 *
 *        Samples are uniformly distributed over the half-open interval *[low, high)*
 *        (includes *low*, but excludes *high*).
 *
 *        Example::
 *
 *        random_uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
 *        [ 0.54488319,  0.84725171]]
 *
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L45
 * \param symbol_name name of the resulting symbol
 * \param low Lower bound of the distribution.
 * \param high Upper bound of the distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_uniform(const std::string& symbol_name,
                             mx_float low = 0,
                             mx_float high = 1,
                             Shape shape = Shape(),
                             const std::string& ctx = "",
                             Random_uniformDtype dtype = Random_uniformDtype::kNone) {
  static const char *Random_uniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_uniformDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_normalDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a normal (Gaussian) distribution.
 *
 *        .. note:: The existing alias ``normal`` is deprecated.
 *
 *        Samples are distributed according to a normal distribution parametrized by
 *
 *        Example::
 *
 *        random_normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
 *        [-1.23474145,  1.55807114]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L62
 * \param symbol_name name of the resulting symbol
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_normal(const std::string& symbol_name,
                            mx_float loc = 0,
                            mx_float scale = 1,
                            Shape shape = Shape(),
                            const std::string& ctx = "",
                            Random_normalDtype dtype = Random_normalDtype::kNone) {
  static const char *Random_normalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_normalDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_gammaDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a gamma distribution.
 *
 *        Samples are distributed according to a gamma distribution parametrized by
 *
 *        Example::
 *
 *        random_gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
 *        [ 3.91697288,  3.65933681]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L75
 * \param symbol_name name of the resulting symbol
 * \param alpha Alpha parameter (shape) of the gamma distribution.
 * \param beta Beta parameter (scale) of the gamma distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_gamma(const std::string& symbol_name,
                           mx_float alpha = 1,
                           mx_float beta = 1,
                           Shape shape = Shape(),
                           const std::string& ctx = "",
                           Random_gammaDtype dtype = Random_gammaDtype::kNone) {
  static const char *Random_gammaDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_gamma")
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_gammaDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_exponentialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from an exponential distribution.
 *
 *        Samples are distributed according to an exponential distribution parametrized
 *
 *        Example::
 *
 *        random_exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
 *        [ 0.04146638,  0.31715935]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L88
 * \param symbol_name name of the resulting symbol
 * \param lam Lambda parameter (rate) of the exponential distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_exponential(const std::string& symbol_name,
                                 mx_float lam = 1,
                                 Shape shape = Shape(),
                                 const std::string& ctx = "",
                                 Random_exponentialDtype dtype = Random_exponentialDtype::kNone) {
  static const char *Random_exponentialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_exponential")
           .SetParam("lam", lam)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_exponentialDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_poissonDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a Poisson distribution.
 *
 *        Samples are distributed according to a Poisson distribution parametrized by
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
 *        [ 4.,  6.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L102
 * \param symbol_name name of the resulting symbol
 * \param lam Lambda parameter (rate) of the Poisson distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_poisson(const std::string& symbol_name,
                             mx_float lam = 1,
                             Shape shape = Shape(),
                             const std::string& ctx = "",
                             Random_poissonDtype dtype = Random_poissonDtype::kNone) {
  static const char *Random_poissonDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_poisson")
           .SetParam("lam", lam)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_poissonDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_negative_binomialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a negative binomial distribution.
 *
 *        Samples are distributed according to a negative binomial distribution
 *        *k* (limit of unsuccessful experiments) and *p* (failure probability in each
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
 *        [ 2.,  5.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L117
 * \param symbol_name name of the resulting symbol
 * \param k Limit of unsuccessful experiments.
 * \param p Failure probability in each experiment.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_negative_binomial(const std::string& symbol_name,
                                       int k = 1,
                                       mx_float p = 1,
                                       Shape shape = Shape(),
                                       const std::string& ctx = "",
                                       Random_negative_binomialDtype dtype = Random_negative_binomialDtype::kNone) {
  static const char *Random_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_negative_binomial")
           .SetParam("k", k)
           .SetParam("p", p)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_negative_binomialDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output in case this can't be inferred. Defaults to float32 if not
 */
enum class Random_generalized_negative_binomialDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \breif Draw random samples from a generalized negative binomial distribution.
 *
 *        Samples are distributed according to a generalized negative binomial
 *        *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is
 *        number of unsuccessful experiments (generalized to real numbers).
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,
 *        [ 6.,  4.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L133
 * \param symbol_name name of the resulting symbol
 * \param mu Mean of the negative binomial distribution.
 * \param alpha Alpha (dispersion) parameter of the negative binomial distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_generalized_negative_binomial(const std::string& symbol_name,
                                                   mx_float mu = 1,
                                                   mx_float alpha = 1,
                                                   Shape shape = Shape(),
                                                   const std::string& ctx = "",
                                                   Random_generalized_negative_binomialDtype dtype = Random_generalized_negative_binomialDtype::kNone) {
  static const char *Random_generalized_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_generalized_negative_binomial")
           .SetParam("mu", mu)
           .SetParam("alpha", alpha)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_generalized_negative_binomialDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate cross entropy of softmax output and one-hot label.
 *
 *        - This operator computes the cross entropy in two steps:
 *        - Applies softmax function on the input array.
 *        - Computes and returns the cross entropy loss between the softmax output and
 *
 *        - The softmax function and cross entropy loss is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        Example::
 *
 *        x = [[1, 2, 3],
 *        [11, 7, 5]]
 *
 *        label = [2, 0]
 *
 *        softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
 *        [0.97962922, 0.01794253, 0.00242826]]
 *
 *        softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) =
 *
 *
 *
 *        Defined in src/operator/loss_binary_op.cc:L40
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in src/operator/swapaxis.cc:L55
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(const std::string& symbol_name,
                       Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Padding type to use. "constant" pads with `constant_value` "edge" pads using
 *        the edge values of the input array "reflect" pads by reflecting values with
 */
enum class PadMode {
  kConstant = 0,
  kEdge = 1,
  kReflect = 2
};

/*!
 * \breif Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,2,2,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in src/operator/pad.cc:L728
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` "edge" pads
 *        using the edge values of the input array "reflect" pads by reflecting values
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(const std::string& symbol_name,
                  Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge",
    "reflect"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is 1.  Specifying -1
 *        axis to be the last item in the input shape.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in src/operator/batch_norm.cc:L396
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param moving_mean running mean of input
 * \param moving_var running variance of input
 * \param eps Epsilon to prevent div 0. Must be bigger than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \param axis Specify which shape axis the channel is specified
 * \param cudnn_off Do not select CUDNN operator, if available
 * \return new symbol
 */
inline Symbol BatchNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        Symbol moving_mean,
                        Symbol moving_var,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false,
                        int axis = 1,
                        bool cudnn_off = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetParam("axis", axis)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .SetInput("moving_mean", moving_mean)
           .SetInput("moving_var", moving_var)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in src/operator/batch_norm_v1.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm_v1(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001,
                           mx_float momentum = 0.9,
                           bool fix_gamma = true,
                           bool use_global_stats = false,
                           bool output_mean_var = false) {
  return Operator("BatchNorm_v1")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * gradient
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L25
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(const std::string& symbol_name,
                         Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L55
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L92
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(const std::string& symbol_name,
                          Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for `RMSProp` optimizer.
 *
 *        `RMSprop` is a variant of stochastic gradient descent where the gradients are
 *        divided by a cache which grows with the sum of squares of recent gradients?
 *
 *        `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
 *        tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate
 *        each parameter monotonically over the course of training.
 *        While this is analytically motivated for convex optimizations, it may not be
 *        for non-convex problems. `RMSProp` deals with this heuristically by allowing the
 *        learning rates to rebound as the denominator decays over time.
 *
 *        Define the Root Mean Square (RMS) error criterion of the gradient as
 *        :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
 *        gradient and :math:`E[g^2]_t` is the decaying average over past squared
 *
 *        The :math:`E[g^2]_t` is given by:
 *
 *        .. math::
 *        E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
 *
 *        The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *        Tieleman & Hinton, 2012.
 *
 *        Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
 *        :math:`\eta` to be 0.001.
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L144
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The decay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for RMSPropAlex optimizer.
 *
 *        `RMSPropAlex` is non-centered version of `RMSProp`.
 *
 *        Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
 *        :math:`E[g]_t` is the decaying average over past gradient.
 *
 *        .. math::
 *        E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
 *        E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
 *        \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 +
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t + \Delta_t
 *
 *        The RMSPropAlex code follows the version in
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *        Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
 *        to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L183
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(const std::string& symbol_name,
                                 Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class LeakyReLUActType {
  kElu = 0,
  kLeaky = 1,
  kPrelu = 2,
  kRrelu = 3
};

/*!
 * \breif Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in src/operator/leaky_relu.cc:L39
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(const std::string& symbol_name,
                        Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::kLeaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(const std::string& symbol_name,
                                        Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif upsampling method
 */
enum class UpSamplingSampleType {
  kBilinear = 0,
  kNearest = 1
};

/*! \breif How to handle multiple input. concat means concatenate upsampled images along
 *        the channel dimension. sum means add all images together, only available for
 */
enum class UpSamplingMultiInputMode {
  kConcat = 0,
  kSum = 1
};

/*!
 * \breif Performs nearest neighbor/bilinear up sampling to inputs.
 * \param symbol_name name of the resulting symbol
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::string& symbol_name,
                         const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::kConcat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will be concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in src/operator/concat.cc:L80
 * \param symbol_name name of the resulting symbol
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        Example::
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in src/operator/slice_channel.cc:L88
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here: http://mxnet.io/how_to/new_op.html.
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data for the custom operator.
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     const std::string& op_type) {
  return Operator("Custom")
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on paper:
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in src/operator/instance_norm.cc:L80
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class PoolingPoolType {
  kAvg = 0,
  kMax = 1,
  kSum = 2
};

/*! \breif Pooling convention to be applied.
 */
enum class PoolingPoolingConvention {
  kFull = 0,
  kValid = 1
};

/*!
 * \breif Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data**: *(batch_size, channel, width)*,
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in src/operator/pooling.cc:L121
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::kValid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in src/operator/crop.cc:L31
 * \param symbol_name name of the resulting symbol
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::string& symbol_name,
                   const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 */
enum class SpatialTransformerTransformType {
  kAffine = 0
};

/*! \breif sampling type
 */
enum class SpatialTransformerSamplerType {
  kBilinear = 0
};

/*!
 * \breif Applies a spatial transformer to input feature map.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(const std::string& symbol_name,
                                 Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algorithm by running performance test.
 */
enum class DeconvolutionCudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \breif Set layout for input, output and weight. Empty for default layout, NCW for 1d,
 */
enum class DeconvolutionLayout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNCW = 3,
  kNDHWC = 4,
  kNHWC = 5
};

/*!
 * \breif Computes 2D transposed convolution (aka fractionally strided convolution) of
 *        the input tensor. This operation can be seen as the gradient of Convolution
 *        operation with respect to its input. Convolution usually reduces the size of
 *        the input. Transposed convolution works the other way, going from a smaller
 * \param symbol_name name of the resulting symbol
 * \param data Input tensor to the deconvolution operation.
 * \param weight Weights representing the kernel.
 * \param bias Bias added to the result after the deconvolution operation.
 * \param kernel Deconvolution kernel size: (h, w) or (d, h, w). This is same as the
 * \param num_filter Number of output filters.
 * \param stride The stride used for the corresponding convolution: (h, w) or (d, h, w).
 * \param dilate Dilation factor for each dimension of the input: (h, w) or (d, h, w).
 * \param pad The amount of implicit zero padding added during convolution for each
 *        dimension of the input: (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good
 *        choice. If `target_shape` is set, `pad` will be ignored and a padding that will
 * \param adj Adjustment for output shape: (h, w) or (d, h, w). If `target_shape` is set,
 * \param target_shape Shape of the output tensor: (h, w) or (d, h, w).
 * \param num_group Number of groups partition.
 * \param workspace Maximum temporal workspace allowed for deconvolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algorithm by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for default layout, NCW
 * \return new symbol
 */
inline Symbol Deconvolution(const std::string& symbol_name,
                            Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::kNone,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::kNone) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalizes the gradient.
 */
enum class SoftmaxOutputNormalization {
  kBatch = 0,
  kNull = 1,
  kValid = 2
};

/*!
 * \breif Computes the gradient of cross entropy loss with respect to softmax output.
 *
 *        - This operator computes the gradient in two steps.
 *        The cross entropy loss does not actually need to be computed.
 *
 *        - Applies softmax function on the input array.
 *        - Computes and returns the gradient of cross entropy loss w.r.t. the softmax
 *
 *        - The softmax function, cross entropy loss and gradient is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        - The gradient of cross entropy loss w.r.t softmax output:
 *
 *        .. math:: \text{gradient} = \text{output} - \text{label}
 *
 *        - During forward propagation, the softmax function is computed for each
 *
 *        For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The
 *        :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters
 *        and `multi_output` to specify the way to compute softmax:
 *
 *        - By default, `preserve_shape` is ``false``. This operator will reshape the
 *        into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the
 *        each row in the reshaped array, and afterwards reshape it back to the original
 *        :math:`(d_1, d_2, ..., d_n)`.
 *        - If `preserve_shape` is ``true``, the softmax function will be computed along
 *        the last axis (`axis` = ``-1``).
 *        - If `multi_output` is ``true``, the softmax function will be computed along
 *        the second axis (`axis` = ``1``).
 *
 *        - During backward propagation, the gradient of cross-entropy loss w.r.t softmax
 *        The provided label can be a one-hot label array or a probability label array.
 *
 *        - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input
 *        with a particular label to be ignored during backward propagation. **This has
 *        softmax `output` has same shape as `label`**.
 *
 *        Example::
 *
 *        data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
 *        label = [1,0,2,3]
 *        ignore_label = 1
 *        SoftmaxOutput(data=data, label = label,\
 *        multi_output=true, use_ignore=true,\
 *        ignore_label=ignore_label)
 *        ## forward softmax output
 *        [[ 0.0320586   0.08714432  0.23688284  0.64391428]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]]
 *        ## backward gradient output
 *        [[ 0.    0.    0.    0.  ]
 *        [-0.75  0.25  0.25  0.25]
 *        [ 0.25  0.25 -0.75  0.25]
 *        [ 0.25  0.25  0.25 -0.75]]
 *        ## notice that the first row is all 0 because label[0] is 1, which is equal to
 *
 *        - The parameter `grad_scale` can be used to rescale the gradient, which is
 *        give each loss function different weights.
 *
 *        - This operator also supports various ways to normalize the gradient by
 *        The `normalization` is applied if softmax output has different shape than the
 *        The `normalization` mode can be set to the followings:
 *
 *        - ``'null'``: do nothing.
 *        - ``'batch'``: divide the gradient by the batch size.
 *        - ``'valid'``: divide the gradient by the number of instances which are not
 *
 *
 *
 *        Defined in src/operator/softmax_output.cc:L108
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param label Ground truth label.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::kNull,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalizes the gradient.
 */
enum class SoftmaxNormalization {
  kBatch = 0,
  kNull = 1,
  kValid = 2
};

/*!
 * \breif Please use `SoftmaxOutput`.
 *
 *        .. note::
 *
 *        This operator has been renamed to `SoftmaxOutput`, which
 *        computes the gradient of cross-entropy loss w.r.t softmax output.
 *        To just compute softmax output, use the `softmax` operator.
 *
 *
 *
 *        Defined in src/operator/softmax_output.cc:L123
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \return new symbol
 */
inline Symbol Softmax(const std::string& symbol_name,
                      Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::kNull,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, y=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, y=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in src/operator/sequence_reverse.cc:L98
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(const std::string& symbol_name,
                              Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies correlation to inputs.
 * \param symbol_name name of the resulting symbol
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(const std::string& symbol_name,
                          Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(const std::string& symbol_name,
                        Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in src/operator/lrn.cc:L58
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(const std::string& symbol_name,
                  Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class Pooling_v1PoolType {
  kAvg = 0,
  kMax = 1,
  kSum = 2
};

/*! \breif Pooling convention to be applied.
 */
enum class Pooling_v1PoolingConvention {
  kFull = 0,
  kValid = 1
};

/*!
 * \breif This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in src/operator/pooling_v1.cc:L85
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(const std::string& symbol_name,
                         Symbol data,
                         Shape kernel,
                         Pooling_v1PoolType pool_type,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::kValid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in src/operator/fully_connected.cc:L74
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, y=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in src/operator/sequence_mask.cc:L112
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in src/operator/sequence_last.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceLast(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \breif The type of transformation. For `affine`, input data should be an affine matrix
 *        of size (batch, 6). For `warp`, input data should be an optical flow of size
 */
enum class GridGeneratorTransformType {
  kAffine = 0,
  kWarp = 1
};

/*!
 * \breif Generates 2D sampling grid for bilinear sampling.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param transform_type The type of transformation. For `affine`, input data should be
 *        an affine matrix of size (batch, 6). For `warp`, input data should be an
 * \param target_shape Specifies the output shape (H, W). This is required if
 *        transformation type is `affine`. If transformation type is `warp`, this
 * \return new symbol
 */
inline Symbol GridGenerator(const std::string& symbol_name,
                            Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 */
enum class Convolution_v1CudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 */
enum class Convolution_v1Layout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNDHWC = 3,
  kNHWC = 4
};

/*!
 * \breif This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::kNone,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::kNone) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class ActivationActType {
  kRelu = 0,
  kSigmoid = 1,
  kSoftrelu = 2,
  kTanh = 3
};

/*!
 * \breif Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *
 *
 *
 *        Defined in src/operator/activation.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data Input array to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(const std::string& symbol_name,
                              Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies dropout operation to input array.
 *
 *        - During training, each element of the input is set to zero with probability p.
 *        The whole array is rescaled by :math:`1/(1-p)` to keep the expected
 *        sum of the input unchanged.
 *
 *        - During testing, this operator does not change the input.
 *
 *        Example::
 *
 *        random.seed(998)
 *        input_array = array([[3., 0.5,  -0.5,  2., 7.],
 *        [2., -0.4,   7.,  3., 0.2]])
 *        a = symbol.Variable('a')
 *        dropout = symbol.Dropout(a, p = 0.2)
 *        executor = dropout.simple_bind(a = input_array.shape)
 *
 *        ## If training
 *        executor.forward(is_train = True, a = input_array)
 *        executor.outputs
 *        [[ 3.75   0.625 -0.     2.5    8.75 ]
 *        [ 2.5   -0.5    8.75   3.75   0.   ]]
 *
 *        ## If testing
 *        executor.forward(is_train = False, a = input_array)
 *        executor.outputs
 *        [[ 3.     0.5   -0.5    2.     7.   ]
 *        [ 2.    -0.4    7.     3.     0.2  ]]
 *
 *
 *        Defined in src/operator/dropout.cc:L62
 * \param symbol_name name of the resulting symbol
 * \param data Input array to which dropout will be applied.
 * \param p Fraction of the input that gets dropped out during training time.
 * \return new symbol
 */
inline Symbol Dropout(const std::string& symbol_name,
                      Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 */
enum class ConvolutionCudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 */
enum class ConvolutionLayout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNCW = 3,
  kNDHWC = 4,
  kNHWC = 5
};

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
 *        width)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concatenating
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in src/operator/convolution.cc:L154
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temporary workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::kNone,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::kNone) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes and optimizes for squared loss during backward propagation.
 *        Just outputs ``data`` during forward propagation.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the squared loss estimated over :math:`n` samples is defined as
 *
 *        :math:`\text{SquaredLoss}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left(
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L51
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(const std::string& symbol_name,
                                     Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i -
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L72
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(const std::string& symbol_name,
                                  Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-x)}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L93
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(const std::string& symbol_name,
                                       Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif If this is set to null, the output gradient will not be normalized. If this is
 *        set to batch, the output gradient will be divided by the batch size. If this is
 *        set to valid, the output gradient will be divided by the number of valid input
 */
enum class MakeLossNormalization {
  kBatch = 0,
  kNull = 1,
  kValid = 2
};

/*!
 * \breif Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in src/operator/make_loss.cc:L52
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::kNull) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in src/operator/roi_pooling.cc:L273
 * \param symbol_name name of the resulting symbol
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(const std::string& symbol_name,
                         Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol(symbol_name);
}

/*! \breif Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 */
enum class SoftmaxActivationMode {
  kChannel = 0,
  kInstance = 1
};

/*!
 * \breif Applies softmax activation to input. This is intended for internal layers.
 *
 *        .. note::
 *
 *        This operator has been deprecated, please use `softmax`.
 *
 *        If `mode` = ``instance``, this operator will compute a softmax for each
 *        This is the default mode.
 *
 *        If `mode` = ``channel``, this operator will compute a k-class softmax at each
 *        of each instance, where `k` = ``num_channel``. This mode can only be used when
 *        has at least 3 dimensions.
 *        This can be used for `fully convolutional network`, `image segmentation`, etc.
 *
 *        Example::
 *
 *        >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
 *        >>>                            [2., -.4, 7.,   3., 0.2]])
 *        >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
 *        >>> print softmax_act.asnumpy()
 *        [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03
 *        [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02
 *
 *
 *
 *        Defined in src/operator/softmax_activation.cc:L48
 * \param symbol_name name of the resulting symbol
 * \param data Input array to activation function.
 * \param mode Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 * \return new symbol
 */
inline Symbol SoftmaxActivation(const std::string& symbol_name,
                                Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::kInstance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Specify the dimension along which to compute L2 norm.
 */
enum class L2NormalizationMode {
  kChannel = 0,
  kInstance = 1,
  kSpatial = 2
};

/*!
 * \breif Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in src/operator/l2_normalization.cc:L74
 * \param symbol_name name of the resulting symbol
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::kInstance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif the type of RNN to compute
 */
enum class RNNMode {
  kGru = 0,
  kLstm = 1,
  kRnn_relu = 2,
  kRnn_tanh = 3
};

/*!
 * \breif Applies a recurrent layer to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(const std::string& symbol_name,
                  Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs. This function assume rhs uses 0-based
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies the softmax function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *
 *        .. math::
 *        softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        Example::
 *
 *        x = [[ 1.  1.  1.]
 *        [ 1.  1.  1.]]
 *
 *        softmax(x,axis=0) = [[ 0.5  0.5  0.5]
 *        [ 0.5  0.5  0.5]]
 *
 *        softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
 *        [ 0.33333334,  0.33333334,  0.33333334]]
 *
 *
 *
 *        Defined in src/operator/nn/softmax.cc:L35
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \return new symbol
 */
inline Symbol softmax(Symbol data,
                      int axis = -1) {
  return Operator("softmax")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the log softmax of the input.
 *        This is equivalent to computing softmax followed by log.
 *
 *        Examples::
 *
 *        >>> x = mx.nd.array([1, 2, .1])
 *        >>> mx.nd.log_softmax(x).asnumpy()
 *        array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
 *
 *        >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
 *        >>> mx.nd.log_softmax(x, axis=0).asnumpy()
 *        array([[-0.34115392, -0.69314718, -1.24115396],
 *        [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
 *
 *
 *
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \return new symbol
 */
inline Symbol log_softmax(Symbol data,
                          int axis = -1) {
  return Operator("log_softmax")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L32
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L71
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L104
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L137
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Reshapes the input array.
 *
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4).The size of the new shape
 *
 *        Example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *
 *        - ``0``  copy this dimension from the input to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *
 *        Example::
 *
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L87
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *
 *        Example::
 *
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L127
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Permutes the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L168
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inserts a new axis of size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L204
 * \param data Source input
 * \param axis Position where new axis is to be inserted. Suppose that the input
 *        `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim,
 * \return new symbol
 */
inline Symbol expand_dims(Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slices a contiguous region of the array.
 *
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *
 *        This function returns a sliced continuous region of the array between the
 *        by `begin` and `end`.
 *
 *        For an input array of `n` dimensions, slice operation with ``begin=(b_0,
 *        and ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape
 *        ``(e_1-b_0, ..., e_n-b_n-1)``.
 *
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        Example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L244
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \return new symbol
 */
inline Symbol slice(Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slices along a given axis.
 *
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *
 *        Examples::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L324
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *        Example::
 *
 *        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
 *        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
 *        dot(x,y)[0,0,1,1] = 0
 *        sum(x[0,0,:]*y[:,1,1]) = 0
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L363
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L399
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Clips (limits) the values in an array.
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_x` would be::
 *
 *        clip(x, a_min, a_max) = max(min(x, a_max), a_min))
 *
 *        Example::
 *
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L444
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeats elements of an array.
 *
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L486
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeats the whole array multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L547
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reverses the order of elements along given axis while preserving array shape.
 *
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in src/operator/tensor/matrix_op.cc:L588
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return an array of zeros with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        zeros_like(x) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 * \param data The input
 * \return new symbol
 */
inline Symbol zeros_like(Symbol data) {
  return Operator("zeros_like")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return an array of ones with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        ones_like(x) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 * \param data The input
 * \return new symbol
 */
inline Symbol ones_like(Symbol data) {
  return Operator("ones_like")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Adds all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *
 *        Defined in src/operator/tensor/elemwise_sum.cc:L63
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol();
}

/*!
 * \breif Returns indices of the maximum values along an axis.
 *
 *        In the case of multiple occurrences of maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmax along axis 0
 *        argmax(x, axis=0) = [ 1.,  1.,  1.]
 *
 *        // argmax along axis 1
 *        argmax(x, axis=1) = [ 2.,  2.]
 *
 *        // argmax along axis 1 keeping same dims as an input array
 *        argmax(x, axis=1, keepdims=True) = [[ 2.],
 *        [ 2.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L33
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmax(Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns indices of the minimum values along an axis.
 *
 *        In the case of multiple occurrences of minimum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmin along axis 0
 *        argmin(x, axis=0) = [ 0.,  0.,  0.]
 *
 *        // argmin along axis 1
 *        argmin(x, axis=1) = [ 0.,  0.]
 *
 *        // argmin along axis 1 keeping same dims as an input array
 *        argmin(x, axis=1, keepdims=True) = [[ 0.],
 *        [ 0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L58
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmin(Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns argmax indices of each channel from the input array.
 *
 *        The result will be an NDArray of shape (num_channel,).
 *
 *        In case of multiple occurrences of the maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        argmax_channel(x) = [ 2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L78
 * \param data The input array
 * \return new symbol
 */
inline Symbol argmax_channel(Symbol data) {
  return Operator("argmax_channel")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Picks elements from an input array according to the input indices along the
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last element along an axis (the `clip` mode).
 *
 *        This function supports n-dimensional input and (n-1)-dimensional indices arrays.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // picks elements with specified indices along axis 0
 *        pick(x, y=[0,1], 0) = [ 1.,  4.]
 *
 *        // picks elements with specified indices along axis 1
 *        pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
 *
 *        y = [[ 1.],
 *        [ 0.],
 *        [ 2.]]
 *
 *        // picks elements with specified indices along axis 1 and dims are maintained
 *        pick(x,y, 1, keepdims=True) = [[ 2.],
 *        [ 3.],
 *        [ 6.]]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L126
 * \param data The input array
 * \param index The index array
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol pick(Symbol data,
                   Symbol index,
                   dmlc::optional<int> axis = dmlc::optional<int>(),
                   bool keepdims = false) {
  return Operator("pick")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .SetInput("index", index)
           .CreateSymbol();
}

/*!
 * \breif Returns result of first array elements raised to powers from second array,
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_power(x, y) = [[ 2.,  2.,  2.],
 *        [ 4.,  4.,  4.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L26
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_power(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise maximum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L61
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_maximum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise minimum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L96
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_minimum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the hypotenuse of a right angled triangle, given its "legs"
 *        with broadcasting.
 *
 *        It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
 *
 *        Example::
 *
 *        x = [[ 3.,  3.,  3.]]
 *
 *        y = [[ 4.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
 *        [ 5.,  5.,  5.]]
 *
 *        z = [[ 0.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
 *        [ 5.,  5.,  5.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L137
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_hypot(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Computes the sum of array elements over given axes.
 *
 *        .. Note::
 *
 *        `sum` and `sum_axis` are equivalent.
 *
 *        Example::
 *
 *        data = [[[1,2],[2,3],[1,3]],
 *        [[1,4],[4,3],[5,2]],
 *        [[7,1],[7,2],[7,3]]]
 *
 *        sum(data, axis=1)
 *        [[  4.   8.]
 *        [ 10.   9.]
 *        [ 21.   6.]]
 *
 *        sum(data, axis=[1,2])
 *        [ 12.  19.  27.]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L51
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol sum(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the mean of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L64
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol mean(Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the product of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L77
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol prod(Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the sum of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L92
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nansum(Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false,
                     bool exclude = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the product of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L107
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nanprod(Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false,
                      bool exclude = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the max of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L121
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol max(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the min of array elements over given axes.
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L135
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol min(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcasts the input array over particular axes.
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        Example::
 *
 *        // given x of shape (1,2,1)
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast x on on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast x on on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L168
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcasts the input array to a new shape.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimension which you do not want to change can also be kept as `0` which
 *        So with `shape=(2,0)`, we will obtain the same result as in the above example.
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L192
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flattens the input array and then computes the l2 norm.
 *
 *        Examples::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        norm(x) = [5.47722578]
 *
 *
 *
 *        Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L218
 * \param data Source input
 * \return new symbol
 */
inline Symbol norm(Symbol data) {
  return Operator("norm")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the top *k* elements in an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // returns an index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // returns the value of top-2 largest elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // returns the value of top-2 smallest elements on last axis
 *        topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
 *        [ 0.1 ,  0.2]]
 *
 *        // returns the value of top-2 largest elements on axis 0
 *        topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
 *        [ 0.1,  0.2,  0.2]]
 *
 *        // flattens and then returns list of both values and indices
 *        topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [
 *
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L44
 * \param data The input array
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 * \param is_ascend Whether to choose k largest or k smallest elements. Top K largest
 * \return new symbol
 */
inline Symbol topk(Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::kIndices,
                   bool is_ascend = false) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns a sorted copy of an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sorts along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flattens and then sorts
 *        sort(x) = [ 1.,  1.,  3.,  4.]
 *
 *        // sorts along the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L107
 * \param data The input array
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the indices that would sort an input array along the given axis.
 *
 *        This function performs sorting along the given axis and returns an array of
 *        as an input array that index data in sorted order.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in src/operator/tensor/ordering_op.cc:L157
 * \param data The input array
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol argsort(Symbol data,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true) {
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Adds arguments element-wise.
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Computes rectified linear.
 *
 *        .. math::
 *        max(features, 0)
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L18
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L36
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L91
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Stops gradient computation.
 *        .. note:: ``make_loss`` is deprecated, use ``MakeLoss``.
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L98
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L155
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Negate src
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:174
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L186
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L201
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L216
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L232
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise ceiling of the input.
 *
 *        The ceil of the scalar x is the smallest integer i, such that i >= x.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L245
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise floor of the input.
 *
 *        The floor of the scalar x is the largest integer i, such that i <= x.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L258
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return the element-wise truncated value of the input.
 *
 *        The truncated value of the scalar x is the nearest integer i which is closer to
 *        zero than x is. In short, the fractional part of the signed number x is
 *
 *        Example::
 *
 *        trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L272
 * \param data The input array.
 * \return new symbol
 */
inline Symbol trunc(Symbol data) {
  return Operator("trunc")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer towards zero of the
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L283
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [3, 9, 16]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L297
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L315
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L333
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [inf, 1, 0.707]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L352
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L362
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L372
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L382
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L398
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L412
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L425
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L441
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L457
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L474
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L491
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L507
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L521
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L535
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L549
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L563
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L577
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic sine of the input array, computed
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L587
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic cosine of the input array, computed
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L597
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic tangent of the input array,
 *
 *
 *        Defined in src/operator/tensor/elemwise_unary_op.cc:L607
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the gamma function (extension of the factorial function to the reals) ,
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:617
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise log of the absolute value of the gamma function of the
 *
 *        From:src/operator/tensor/elemwise_unary_op.cc:627
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Maps integer indices to vector representations (embeddings).
 *
 *        This operator maps words to real-valued vectors in a high-dimensional space,
 *        called word embeddings. These embeddings can capture semantic and syntactic
 *        For example, it has been noted that in the learned embedding spaces, similar
 *        to be close to each other and dissimilar words far apart.
 *
 *        For an input array of shape (d1, ..., dK),
 *        the shape of an output array is (d1, ..., dK, output_dim).
 *        All the input values should be integers in the range [0, input_dim).
 *
 *        If the input_dim is ip0 and output_dim is op0, then shape of the embedding
 *        (ip0, op0).
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last vector in an embedding matrix.
 *
 *        Examples::
 *
 *        input_dim = 4
 *        output_dim = 5
 *
 *        // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
 *        y = [[  0.,   1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.,   9.],
 *        [ 10.,  11.,  12.,  13.,  14.],
 *        [ 15.,  16.,  17.,  18.,  19.]]
 *
 *        // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
 *        x = [[ 1.,  3.],
 *        [ 0.,  2.]]
 *
 *        // Mapped input x to its vector representation y.
 *        Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
 *        [ 15.,  16.,  17.,  18.,  19.]],
 *
 *        [[  0.,   1.,   2.,   3.,   4.],
 *        [ 10.,  11.,  12.,  13.,  14.]]]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L55
 * \param data The input array to the embedding operator.
 * \param weight The embedding weight matrix.
 * \param input_dim Vocabulary size of the input indices.
 * \param output_dim Dimension of the embedding vectors.
 * \param dtype Data type of weight.
 * \return new symbol
 */
inline Symbol Embedding(Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim,
                        EmbeddingDtype dtype = EmbeddingDtype::kFloat32) {
  static const char *EmbeddingDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetParam("dtype", EmbeddingDtypeValues[int(dtype)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol();
}

/*!
 * \breif Takes elements from an input array along the given axis.
 *
 *        This function slices the input array along a particular axis with the provided
 *
 *        Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0,
 *        will have shape ``(i0, i1, d1, d2)``, computed by::
 *
 *        output[i,j,:,:] = input[indices[i,j],:,:]
 *
 *        .. note::
 *        - `axis`- Only slicing along axis 0 is supported for now.
 *        - `mode`- Only `clip` mode is supported for now.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices along axis 0
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L117
 * \param a The input array.
 * \param indices The indices of the values to be extracted.
 * \param axis The axis of input array to be taken.
 * \param mode Specify how out-of-bound indices bahave. "clip" means clip to the range.
 *        So, if all indices mentioned are too large, they are replaced by the index that
 *        addresses the last element along an axis.  "wrap" means to wrap around.
 * \return new symbol
 */
inline Symbol take(Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::kClip) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Takes elements from a data batch.
 *
 *        .. note::
 *        `batch_take` is deprecated. Use `pick` instead.
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L172
 * \param a The input array
 * \param indices The index array
 * \return new symbol
 */
inline Symbol batch_take(Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Returns a one-hot array.
 *
 *        The locations represented by `indices` take value `on_value`, while all
 *        other locations take value `off_value`.
 *
 *        `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d``
 *        in an output array of shape ``(i0, i1, d)`` with::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in src/operator/tensor/indexing_op.cc:L218
 * \param indices array of locations where to set on_value
 * \param depth Depth of the one hot dimension.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::kFloat32) {
  static const char *One_hotDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **equal to** (==) comparison operation with
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L27
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_equal(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **not equal to** (!=) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L45
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_not_equal(Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **greater than** (>) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L63
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **greater than or equal to** (>=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L81
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **lesser than** (<) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L99
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser(Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns the result of element-wise **lesser than or equal to** (<=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L117
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Performs general matrix multiplication and accumulation.
 *        Input are three tensors *A*, *B*, *C* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\ , *C*\ :sub:`i` be the matrices given by the
 *        The operator performs the BLAS3 function *gemm*
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ )
 *
 *        on all such triples of matrices. Here *alpha* and *beta* are scalar operator
 *        is either the identity or the matrix transposition.
 *
 *        In case of *n=2*, a single *gemm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply-add
 *        A = [[1.0, 1.0], [1.0, 1.0]]
 *        B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
 *        C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *        linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
 *        = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]
 *
 *        // Batch matrix multiply-add
 *        A = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        B = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        C = [[[10.0]], [[0.01]]]
 *        linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
 *        = [[[104.0]], [[0.14]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L48
 * \param A Tensor of input matrices
 * \param B Tensor of input matrices
 * \param C Tensor of input matrices
 * \param transpose_a Multiply with transposed of first input (A).
 * \param transpose_b Multiply with transposed of second input (B).
 * \param alpha Scalar factor multiplied with A*B.
 * \param beta Scalar factor multiplied with C.
 * \return new symbol
 */
inline Symbol linalg_gemm(Symbol A,
                          Symbol B,
                          Symbol C,
                          bool transpose_a = false,
                          bool transpose_b = false,
                          double alpha = 1,
                          double beta = 1) {
  return Operator("linalg_gemm")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetInput("A", A)
           .SetInput("B", B)
           .SetInput("C", C)
           .CreateSymbol();
}

/*!
 * \breif Performs general matrix multiplication.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *gemm* (restricted to two arguments)
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ )
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter and
 *        the identity or the matrix transposition.
 *
 *        In case of *n=2*, a single *gemm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply
 *        A = [[1.0, 1.0], [1.0, 1.0]]
 *        B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
 *        linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0)
 *        = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
 *
 *        // Batch matrix multiply
 *        A = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        B = [[[1.0, 1.0]], [[0.1, 0.1]]]
 *        linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0 )
 *        = [[[4.0]], [[0.04 ]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L106
 * \param A Tensor of input matrices
 * \param B Tensor of input matrices
 * \param transpose_a Multiply with transposed of first input (A).
 * \param transpose_b Multiply with transposed of second input (B).
 * \param alpha Scalar factor multiplied with A*B.
 * \return new symbol
 */
inline Symbol linalg_gemm2(Symbol A,
                           Symbol B,
                           bool transpose_a = false,
                           bool transpose_b = false,
                           double alpha = 1) {
  return Operator("linalg_gemm2")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol();
}

/*!
 * \breif Performs Cholesky factorization of a symmetric positive-definite matrix.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator performs the Cholesky factorization (LAPACK function *potrf*)
 *        on each *A*\ :sub:`i`\ ,
 *        i.e. it computes a lower triangular matrix *U*\ :sub:`i` such that
 *
 *        *A*\ :sub:`i`\  = *U*\ :sub:`i`\  \* *U*\ :sub:`i`\ \ :sup:`T`
 *
 *        for all such matrices. The matrices *A*\ :sub:`i` must be all symmetric and
 *        The resulting matrices *U*\ :sub:`i` will contain zeros in the upper triangle
 *        apart from the diagonal.
 *
 *        In case of *n=2*, a single Cholesky factorization is performed on the matrix
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix factorization
 *        A = [[4.0, 1.0], [1.0, 4.25]]
 *        linalg_potrf(A) = [[2.0, 0], [0.5, 2.0]]
 *
 *        // Batch matrix factorization
 *        A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
 *        linalg_potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L159
 * \param A Tensor of input matrices to be decomposed
 * \return new symbol
 */
inline Symbol linalg_potrf(Symbol A) {
  return Operator("linalg_potrf")
           .SetInput("A", A)
           .CreateSymbol();
}

/*!
 * \breif Performs matrix inversion from a Cholesky factorization.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator assumes that each *A*\ :sub:`i` is the Cholesky factorization of
 *        positive-definite matrix *B*\ :sub:`i` given as a lower triangular matrix
 *        (so *A* is the output of a prior call to operator *linalg_potrf*). The operator
 *        inverse of each *B*\ :sub:`i` from this decomposition, i.e
 *
 *        *out*\ :sub:`i` = *B*\ :sub:`i`\ \ :sup:`-1`
 *
 *        for all such matrices.
 *
 *        In case of *n=2*, the operation is performed on the matrix *A* itself.
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix inverse
 *        A = [[2.0, 0], [0.5, 2.0]]
 *        linalg_potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]
 *
 *        // Batch matrix inverse
 *        A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
 *        linalg_potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
 *        [[0.06641, -0.01562], [-0.01562, 0,0625]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L211
 * \param A Tensor of lower triangular matrices
 * \return new symbol
 */
inline Symbol linalg_potri(Symbol A) {
  return Operator("linalg_potri")
           .SetInput("A", A)
           .CreateSymbol();
}

/*!
 * \breif Performs multiplication with a triangular matrix.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *trmm*
 *
 *        *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *B*\ :sub:`i`
 *
 *        or
 *
 *        *out*\ :sub:`i` = *alpha* \* *B*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ )
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter,
 *        the identity or the matrix transposition (depending on the parameter
 *        order of matrix multiplication depends on the parameter *rightside*.
 *        All matrices *A*\ :sub:`i` must be lower triangular.
 *
 *        In case of *n=2*, a single *trmm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix multiply
 *        A = [[1.0, 0], [1.0, 1.0]]
 *        B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *        linalg_trmm(A, B, alpha = 2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
 *
 *        // Batch matrix multiply
 *        A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
 *        B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
 *        linalg_trmm(A, B, alpha = 2.0 ) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
 *        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
 *
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L268
 * \param A Tensor of lower triangular matrices
 * \param B Tensor of matrices
 * \param transpose Use transposed of the triangular matrix
 * \param rightside Multiply triangular matrix from the right to non-triangular one.
 * \param alpha Scalar factor to be applied to the result.
 * \return new symbol
 */
inline Symbol linalg_trmm(Symbol A,
                          Symbol B,
                          bool transpose = false,
                          bool rightside = false,
                          double alpha = 1) {
  return Operator("linalg_trmm")
           .SetParam("transpose", transpose)
           .SetParam("rightside", rightside)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol();
}

/*!
 * \breif Solves matrix equations involving a triangular matrix.
 *        Input are two tensors *A*, *B* each of dimension *n >= 2* and each
 *        having the same shape on the leading *n-2* dimensions. For every *n-2*
 *        *A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2*
 *        The operator performs the BLAS3 function *trsm*, i.e. it solves the equation
 *
 *        *op*\ (*A*\ :sub:`i`\ ) \* *X*\ :sub:`i` = *alpha* \* *B*\ :sub:`i`
 *
 *        or
 *
 *        *X*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ ) = *alpha* \* *B*\ :sub:`i`
 *
 *        on all such pairs of matrices. Here *alpha* is a scalar operator parameter,
 *        the identity or the matrix transposition (depending on the parameter
 *        order of multiplication on the left depends on the parameter *rightside*.
 *        All matrices *A*\ :sub:`i` must be lower triangular.
 *
 *        In case of *n=2*, a single *trsm* function is performed on the matrices *A*,
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix solve
 *        A = [[1.0, 0], [1.0, 1.0]]
 *        B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
 *        linalg_trsm(A, B, alpha = 0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
 *
 *        // Batch matrix solve
 *        A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
 *        B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
 *        [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
 *        linalg_trsm(A, B, alpha = 0.5 ) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
 *        [[2.0, 2.0, 2.0 ], [2.0, 2.0, 2.0]]]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L331
 * \param A Tensor of lower triangular matrices
 * \param B Tensor of matrices
 * \param transpose Use transposed of the triangular matrix
 * \param rightside Multiply triangular matrix from the right to non-triangular one.
 * \param alpha Scalar factor to be applied to the result.
 * \return new symbol
 */
inline Symbol linalg_trsm(Symbol A,
                          Symbol B,
                          bool transpose = false,
                          bool rightside = false,
                          double alpha = 1) {
  return Operator("linalg_trsm")
           .SetParam("transpose", transpose)
           .SetParam("rightside", rightside)
           .SetParam("alpha", alpha)
           .SetInput("A", A)
           .SetInput("B", B)
           .CreateSymbol();
}

/*!
 * \breif Computes the sum of the logarithms of all diagonal elements in a matrix.
 *        Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index
 *        *A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
 *        The operator performs a reduction of each such matrix to a scalar by summing up
 *        of all diagonal elements. All matrices must be square and all diagonal elements
 *
 *        In case of *n=2*, *A* represents a single matrix on which the reduction will be
 *
 *        .. note:: The operator does only support float32 and float64 data types and
 *        proper backward gradients.
 *
 *        Examples::
 *
 *        // Single matrix reduction
 *        A = [[1.0, 1.0], [1.0, 7.0]]
 *        linalg_sumlogdiag(A) = [1.9459]
 *
 *        // Batch matrix reduction
 *        A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
 *        linalg_sumlogdiag(A) = [1.9459, 3.9318]
 *
 *
 *        Defined in src/operator/tensor/la_op.cc:L379
 * \param A Tensor of square matrices
 * \return new symbol
 */
inline Symbol linalg_sumlogdiag(Symbol A) {
  return Operator("linalg_sumlogdiag")
           .SetInput("A", A)
           .CreateSymbol();
}

/*!
 * \breif Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        from x or y, depending on the elements from condition are true or false. x and
 *        y must have the same shape. If condition has the same shape as x, each element
 *        in the output array is from x if the corresponding element in the condition is
 *        true, and from y if false. If condition does not have the same shape as x, it
 *        must be a 1D array whose size is the same as x's first dimension size. Each row
 *        of the output array is from x's row if the corresponding element from condition
 *
 *        From:src/operator/tensor/control_flow_op.cc:21
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol();
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar) by summing
 *
 *        .. math::
 *
 *        f(x) =
 *        \begin{cases}
 *        (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
 *        |x|-0.5/\sigma^2,& \text{otherwise}
 *        \end{cases}
 *
 *        where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the
 *
 *        Example::
 *
 *        smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]
 *
 *
 *
 *        Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L79
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple multinomial distributions.
 *
 *        *data* is an *n* dimensional array whose last dimension has length *k*, where
 *        *k* is the number of possible outcomes of each multinomial distribution. This
 *        operator will draw *shape* samples from each distribution. If shape is empty
 *        one sample will be drawn from each distribution.
 *
 *        If *get_prob* is true, a second array containing log likelihood of the drawn
 *        samples will also be returned. This is usually used for reinforcement learning
 *        where you can provide reward as head gradient for this array to estimate
 *        gradient.
 *
 *        Note that the input distribution must be normalized, i.e. *data* must sum to
 *        1 along its last axis.
 *
 *        Examples::
 *
 *        probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]
 *
 *        // Draw a single sample for each distribution
 *        sample_multinomial(probs) = [3, 0]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_multinomial(probs, shape=(2)) = [[4, 2],
 *        [0, 0]]
 *
 *        // requests log likelihood
 *        sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
 *
 * \param data Distribution probabilities. Must sum to one on the last axis.
 * \param shape Shape to be sampled from each random distribution.
 * \param get_prob Whether to also return the log probability of sampled result. This is
 *        usually used for differentiating through stochastic variables, e.g. in
 * \param dtype DType of the output in case this can't be inferred. Only support int32
 * \return new symbol
 */
inline Symbol sample_multinomial(Symbol data,
                                 Shape shape = Shape(),
                                 bool get_prob = false,
                                 Sample_multinomialDtype dtype = Sample_multinomialDtype::kInt32) {
  static const char *Sample_multinomialDtypeValues[] = {
    "int32"
  };
  return Operator("sample_multinomial")
           .SetParam("shape", shape)
           .SetParam("get_prob", get_prob)
           .SetParam("dtype", Sample_multinomialDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        uniform distributions on the intervals given by *[low,high)*.
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        low = [ 0.0, 2.5 ]
 *        high = [ 1.0, 3.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_uniform(low, high) = [ 0.40451524,  3.18687344]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
 *        [ 3.18687344,  3.68352246]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L363
 * \param low Lower bounds of the distributions.
 * \param high Upper bounds of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_uniform(Symbol low,
                             Symbol high,
                             Shape shape = Shape(),
                             Sample_uniformDtype dtype = Sample_uniformDtype::kNone) {
  static const char *Sample_uniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_uniform")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_uniformDtypeValues[int(dtype)])
           .SetInput("low", low)
           .SetInput("high", high)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        normal distributions with parameters *mu* (mean) and *sigma* (standard
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        mu = [ 0.0, 2.5 ]
 *        sigma = [ 1.0, 3.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_normal(mu, sigma) = [-0.56410581,  0.95934606]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
 *        [ 0.95934606,  4.48287058]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L365
 * \param mu Means of the distributions.
 * \param sigma Standard deviations of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_normal(Symbol mu,
                            Symbol sigma,
                            Shape shape = Shape(),
                            Sample_normalDtype dtype = Sample_normalDtype::kNone) {
  static const char *Sample_normalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_normal")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_normalDtypeValues[int(dtype)])
           .SetInput("mu", mu)
           .SetInput("sigma", sigma)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        gamma distributions with parameters *alpha* (shape) and *beta* (scale).
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Examples::
 *
 *        alpha = [ 0.0, 2.5 ]
 *        beta = [ 1.0, 0.7 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
 *        [ 2.25797319,  1.70734084]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L368
 * \param alpha Alpha (shape) parameters of the distributions.
 * \param beta Beta (scale) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_gamma(Symbol alpha,
                           Symbol beta,
                           Shape shape = Shape(),
                           Sample_gammaDtype dtype = Sample_gammaDtype::kNone) {
  static const char *Sample_gammaDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_gamma")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_gammaDtypeValues[int(dtype)])
           .SetInput("alpha", alpha)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        exponential distributions with parameters lambda (rate).
 *
 *        The parameters of the distributions are provided as an input array.
 *        Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input array,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input value at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input array.
 *
 *        Examples::
 *
 *        lam = [ 1.0, 8.5 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_exponential(lam) = [ 0.51837951,  0.09994757]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
 *        [ 0.09994757,  0.50447971]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L370
 * \param lam Lambda (rate) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_exponential(Symbol lam,
                                 Shape shape = Shape(),
                                 Sample_exponentialDtype dtype = Sample_exponentialDtype::kNone) {
  static const char *Sample_exponentialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_exponential")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_exponentialDtypeValues[int(dtype)])
           .SetInput("lam", lam)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        Poisson distributions with parameters lambda (rate).
 *
 *        The parameters of the distributions are provided as an input array.
 *        Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input array,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input value at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input array.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        lam = [ 1.0, 8.5 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_poisson(lam) = [  0.,  13.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_poisson(lam, shape=(2)) = [[  0.,   4.],
 *        [ 13.,   8.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L372
 * \param lam Lambda (rate) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_poisson(Symbol lam,
                             Shape shape = Shape(),
                             Sample_poissonDtype dtype = Sample_poissonDtype::kNone) {
  static const char *Sample_poissonDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_poisson")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_poissonDtypeValues[int(dtype)])
           .SetInput("lam", lam)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        negative binomial distributions with parameters *k* (failure limit) and *p*
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        k = [ 20, 49 ]
 *        p = [ 0.4 , 0.77 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_negative_binomial(k, p) = [ 15.,  16.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
 *        [ 16.,  12.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L375
 * \param k Limits of unsuccessful experiments.
 * \param p Failure probabilities in each experiment.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_negative_binomial(Symbol k,
                                       Symbol p,
                                       Shape shape = Shape(),
                                       Sample_negative_binomialDtype dtype = Sample_negative_binomialDtype::kNone) {
  static const char *Sample_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_negative_binomial")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_negative_binomialDtypeValues[int(dtype)])
           .SetInput("k", k)
           .SetInput("p", p)
           .CreateSymbol();
}

/*!
 * \breif Concurrent sampling from multiple
 *        generalized negative binomial distributions with parameters *mu* (mean) and
 *
 *        The parameters of the distributions are provided as input arrays.
 *        Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
 *        be the shape specified as the parameter of the operator, and *m* be the
 *        of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape
 *
 *        For any valid *n*-dimensional index *i* with respect to the input arrays,
 *        will be an *m*-dimensional array that holds randomly drawn samples from the
 *        which is parameterized by the input values at index *i*. If the shape parameter
 *        operator is not set, then one sample will be drawn per distribution and the
 *        has the same shape as the input arrays.
 *
 *        Samples will always be returned as a floating point data type.
 *
 *        Examples::
 *
 *        mu = [ 2.0, 2.5 ]
 *        alpha = [ 1.0, 0.1 ]
 *
 *        // Draw a single sample for each distribution
 *        sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]
 *
 *        // Draw a vector containing two samples for each distribution
 *        sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
 *        [ 3.,  1.]]
 *
 *
 *        Defined in src/operator/random/multisample_op.cc:L379
 * \param mu Means of the distributions.
 * \param alpha Alpha (dispersion) parameters of the distributions.
 * \param shape Shape to be sampled from each random distribution.
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol sample_generalized_negative_binomial(Symbol mu,
                                                   Symbol alpha,
                                                   Shape shape = Shape(),
                                                   Sample_generalized_negative_binomialDtype dtype = Sample_generalized_negative_binomialDtype::kNone) {
  static const char *Sample_generalized_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("sample_generalized_negative_binomial")
           .SetParam("shape", shape)
           .SetParam("dtype", Sample_generalized_negative_binomialDtypeValues[int(dtype)])
           .SetInput("mu", mu)
           .SetInput("alpha", alpha)
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a uniform distribution.
 *
 *        .. note:: The existing alias ``uniform`` is deprecated.
 *
 *        Samples are uniformly distributed over the half-open interval *[low, high)*
 *        (includes *low*, but excludes *high*).
 *
 *        Example::
 *
 *        random_uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
 *        [ 0.54488319,  0.84725171]]
 *
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L45
 * \param low Lower bound of the distribution.
 * \param high Upper bound of the distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_uniform(mx_float low = 0,
                             mx_float high = 1,
                             Shape shape = Shape(),
                             const std::string& ctx = "",
                             Random_uniformDtype dtype = Random_uniformDtype::kNone) {
  static const char *Random_uniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_uniformDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a normal (Gaussian) distribution.
 *
 *        .. note:: The existing alias ``normal`` is deprecated.
 *
 *        Samples are distributed according to a normal distribution parametrized by
 *
 *        Example::
 *
 *        random_normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
 *        [-1.23474145,  1.55807114]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L62
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_normal(mx_float loc = 0,
                            mx_float scale = 1,
                            Shape shape = Shape(),
                            const std::string& ctx = "",
                            Random_normalDtype dtype = Random_normalDtype::kNone) {
  static const char *Random_normalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_normalDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a gamma distribution.
 *
 *        Samples are distributed according to a gamma distribution parametrized by
 *
 *        Example::
 *
 *        random_gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
 *        [ 3.91697288,  3.65933681]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L75
 * \param alpha Alpha parameter (shape) of the gamma distribution.
 * \param beta Beta parameter (scale) of the gamma distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_gamma(mx_float alpha = 1,
                           mx_float beta = 1,
                           Shape shape = Shape(),
                           const std::string& ctx = "",
                           Random_gammaDtype dtype = Random_gammaDtype::kNone) {
  static const char *Random_gammaDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_gamma")
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_gammaDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from an exponential distribution.
 *
 *        Samples are distributed according to an exponential distribution parametrized
 *
 *        Example::
 *
 *        random_exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
 *        [ 0.04146638,  0.31715935]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L88
 * \param lam Lambda parameter (rate) of the exponential distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_exponential(mx_float lam = 1,
                                 Shape shape = Shape(),
                                 const std::string& ctx = "",
                                 Random_exponentialDtype dtype = Random_exponentialDtype::kNone) {
  static const char *Random_exponentialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_exponential")
           .SetParam("lam", lam)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_exponentialDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a Poisson distribution.
 *
 *        Samples are distributed according to a Poisson distribution parametrized by
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
 *        [ 4.,  6.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L102
 * \param lam Lambda parameter (rate) of the Poisson distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_poisson(mx_float lam = 1,
                             Shape shape = Shape(),
                             const std::string& ctx = "",
                             Random_poissonDtype dtype = Random_poissonDtype::kNone) {
  static const char *Random_poissonDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_poisson")
           .SetParam("lam", lam)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_poissonDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a negative binomial distribution.
 *
 *        Samples are distributed according to a negative binomial distribution
 *        *k* (limit of unsuccessful experiments) and *p* (failure probability in each
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
 *        [ 2.,  5.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L117
 * \param k Limit of unsuccessful experiments.
 * \param p Failure probability in each experiment.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_negative_binomial(int k = 1,
                                       mx_float p = 1,
                                       Shape shape = Shape(),
                                       const std::string& ctx = "",
                                       Random_negative_binomialDtype dtype = Random_negative_binomialDtype::kNone) {
  static const char *Random_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_negative_binomial")
           .SetParam("k", k)
           .SetParam("p", p)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_negative_binomialDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a generalized negative binomial distribution.
 *
 *        Samples are distributed according to a generalized negative binomial
 *        *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is
 *        number of unsuccessful experiments (generalized to real numbers).
 *        Samples will always be returned as a floating point data type.
 *
 *        Example::
 *
 *        random_generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,
 *        [ 6.,  4.]]
 *
 *
 *        Defined in src/operator/random/sample_op.cc:L133
 * \param mu Mean of the negative binomial distribution.
 * \param alpha Alpha (dispersion) parameter of the negative binomial distribution.
 * \param shape Shape of the output.
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for
 * \param dtype DType of the output in case this can't be inferred. Defaults to float32
 * \return new symbol
 */
inline Symbol random_generalized_negative_binomial(mx_float mu = 1,
                                                   mx_float alpha = 1,
                                                   Shape shape = Shape(),
                                                   const std::string& ctx = "",
                                                   Random_generalized_negative_binomialDtype dtype = Random_generalized_negative_binomialDtype::kNone) {
  static const char *Random_generalized_negative_binomialDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("random_generalized_negative_binomial")
           .SetParam("mu", mu)
           .SetParam("alpha", alpha)
           .SetParam("shape", shape)
           .SetParam("dtype", Random_generalized_negative_binomialDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Calculate cross entropy of softmax output and one-hot label.
 *
 *        - This operator computes the cross entropy in two steps:
 *        - Applies softmax function on the input array.
 *        - Computes and returns the cross entropy loss between the softmax output and
 *
 *        - The softmax function and cross entropy loss is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        Example::
 *
 *        x = [[1, 2, 3],
 *        [11, 7, 5]]
 *
 *        label = [2, 0]
 *
 *        softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
 *        [0.97962922, 0.01794253, 0.00242826]]
 *
 *        softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) =
 *
 *
 *
 *        Defined in src/operator/loss_binary_op.cc:L40
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in src/operator/swapaxis.cc:L55
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,2,2,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in src/operator/pad.cc:L728
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` "edge" pads
 *        using the edge values of the input array "reflect" pads by reflecting values
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge",
    "reflect"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is 1.  Specifying -1
 *        axis to be the last item in the input shape.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in src/operator/batch_norm.cc:L396
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param moving_mean running mean of input
 * \param moving_var running variance of input
 * \param eps Epsilon to prevent div 0. Must be bigger than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \param axis Specify which shape axis the channel is specified
 * \param cudnn_off Do not select CUDNN operator, if available
 * \return new symbol
 */
inline Symbol BatchNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        Symbol moving_mean,
                        Symbol moving_var,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false,
                        int axis = 1,
                        bool cudnn_off = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetParam("axis", axis)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .SetInput("moving_mean", moving_mean)
           .SetInput("moving_var", moving_var)
           .CreateSymbol();
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in src/operator/batch_norm_v1.cc:L71
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm_v1(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001,
                           mx_float momentum = 0.9,
                           bool fix_gamma = true,
                           bool use_global_stats = false,
                           bool output_mean_var = false) {
  return Operator("BatchNorm_v1")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * gradient
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L25
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol();
}

/*!
 * \breif Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L55
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol();
}

/*!
 * \breif Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L92
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol();
}

/*!
 * \breif Update function for `RMSProp` optimizer.
 *
 *        `RMSprop` is a variant of stochastic gradient descent where the gradients are
 *        divided by a cache which grows with the sum of squares of recent gradients?
 *
 *        `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
 *        tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate
 *        each parameter monotonically over the course of training.
 *        While this is analytically motivated for convex optimizations, it may not be
 *        for non-convex problems. `RMSProp` deals with this heuristically by allowing the
 *        learning rates to rebound as the denominator decays over time.
 *
 *        Define the Root Mean Square (RMS) error criterion of the gradient as
 *        :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
 *        gradient and :math:`E[g^2]_t` is the decaying average over past squared
 *
 *        The :math:`E[g^2]_t` is given by:
 *
 *        .. math::
 *        E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
 *
 *        The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *        Tieleman & Hinton, 2012.
 *
 *        Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
 *        :math:`\eta` to be 0.001.
 *
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L144
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The decay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol();
}

/*!
 * \breif Update function for RMSPropAlex optimizer.
 *
 *        `RMSPropAlex` is non-centered version of `RMSProp`.
 *
 *        Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
 *        :math:`E[g]_t` is the decaying average over past gradient.
 *
 *        .. math::
 *        E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
 *        E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
 *        \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 +
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t + \Delta_t
 *
 *        The RMSPropAlex code follows the version in
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *        Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
 *        to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
 *
 *
 *        Defined in src/operator/optimizer_op.cc:L183
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol();
}

/*!
 * \breif Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in src/operator/leaky_relu.cc:L39
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::kLeaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Performs nearest neighbor/bilinear up sampling to inputs.
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::kConcat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol();
}

/*!
 * \breif Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will be concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in src/operator/concat.cc:L80
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol();
}

/*!
 * \breif Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        Example::
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in src/operator/slice_channel.cc:L88
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here: http://mxnet.io/how_to/new_op.html.
 *
 *
 * \param data Input data for the custom operator.
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \return new symbol
 */
inline Symbol Custom(const std::vector<Symbol>& data,
                     const std::string& op_type) {
  return Operator("Custom")
(data)
           .CreateSymbol();
}

/*!
 * \breif Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on paper:
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in src/operator/instance_norm.cc:L80
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data**: *(batch_size, channel, width)*,
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in src/operator/pooling.cc:L121
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::kValid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in src/operator/crop.cc:L31
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol();
}

/*!
 * \breif Applies a spatial transformer to input feature map.
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol();
}

/*!
 * \breif Computes 2D transposed convolution (aka fractionally strided convolution) of
 *        the input tensor. This operation can be seen as the gradient of Convolution
 *        operation with respect to its input. Convolution usually reduces the size of
 *        the input. Transposed convolution works the other way, going from a smaller
 * \param data Input tensor to the deconvolution operation.
 * \param weight Weights representing the kernel.
 * \param bias Bias added to the result after the deconvolution operation.
 * \param kernel Deconvolution kernel size: (h, w) or (d, h, w). This is same as the
 * \param num_filter Number of output filters.
 * \param stride The stride used for the corresponding convolution: (h, w) or (d, h, w).
 * \param dilate Dilation factor for each dimension of the input: (h, w) or (d, h, w).
 * \param pad The amount of implicit zero padding added during convolution for each
 *        dimension of the input: (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good
 *        choice. If `target_shape` is set, `pad` will be ignored and a padding that will
 * \param adj Adjustment for output shape: (h, w) or (d, h, w). If `target_shape` is set,
 * \param target_shape Shape of the output tensor: (h, w) or (d, h, w).
 * \param num_group Number of groups partition.
 * \param workspace Maximum temporal workspace allowed for deconvolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algorithm by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for default layout, NCW
 * \return new symbol
 */
inline Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::kNone,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::kNone) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Computes the gradient of cross entropy loss with respect to softmax output.
 *
 *        - This operator computes the gradient in two steps.
 *        The cross entropy loss does not actually need to be computed.
 *
 *        - Applies softmax function on the input array.
 *        - Computes and returns the gradient of cross entropy loss w.r.t. the softmax
 *
 *        - The softmax function, cross entropy loss and gradient is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        - The gradient of cross entropy loss w.r.t softmax output:
 *
 *        .. math:: \text{gradient} = \text{output} - \text{label}
 *
 *        - During forward propagation, the softmax function is computed for each
 *
 *        For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The
 *        :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters
 *        and `multi_output` to specify the way to compute softmax:
 *
 *        - By default, `preserve_shape` is ``false``. This operator will reshape the
 *        into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the
 *        each row in the reshaped array, and afterwards reshape it back to the original
 *        :math:`(d_1, d_2, ..., d_n)`.
 *        - If `preserve_shape` is ``true``, the softmax function will be computed along
 *        the last axis (`axis` = ``-1``).
 *        - If `multi_output` is ``true``, the softmax function will be computed along
 *        the second axis (`axis` = ``1``).
 *
 *        - During backward propagation, the gradient of cross-entropy loss w.r.t softmax
 *        The provided label can be a one-hot label array or a probability label array.
 *
 *        - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input
 *        with a particular label to be ignored during backward propagation. **This has
 *        softmax `output` has same shape as `label`**.
 *
 *        Example::
 *
 *        data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
 *        label = [1,0,2,3]
 *        ignore_label = 1
 *        SoftmaxOutput(data=data, label = label,\
 *        multi_output=true, use_ignore=true,\
 *        ignore_label=ignore_label)
 *        ## forward softmax output
 *        [[ 0.0320586   0.08714432  0.23688284  0.64391428]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]]
 *        ## backward gradient output
 *        [[ 0.    0.    0.    0.  ]
 *        [-0.75  0.25  0.25  0.25]
 *        [ 0.25  0.25 -0.75  0.25]
 *        [ 0.25  0.25  0.25 -0.75]]
 *        ## notice that the first row is all 0 because label[0] is 1, which is equal to
 *
 *        - The parameter `grad_scale` can be used to rescale the gradient, which is
 *        give each loss function different weights.
 *
 *        - This operator also supports various ways to normalize the gradient by
 *        The `normalization` is applied if softmax output has different shape than the
 *        The `normalization` mode can be set to the followings:
 *
 *        - ``'null'``: do nothing.
 *        - ``'batch'``: divide the gradient by the batch size.
 *        - ``'valid'``: divide the gradient by the number of instances which are not
 *
 *
 *
 *        Defined in src/operator/softmax_output.cc:L108
 * \param data Input array.
 * \param label Ground truth label.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \return new symbol
 */
inline Symbol SoftmaxOutput(Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::kNull,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Please use `SoftmaxOutput`.
 *
 *        .. note::
 *
 *        This operator has been renamed to `SoftmaxOutput`, which
 *        computes the gradient of cross-entropy loss w.r.t softmax output.
 *        To just compute softmax output, use the `softmax` operator.
 *
 *
 *
 *        Defined in src/operator/softmax_output.cc:L123
 * \param data Input array.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \return new symbol
 */
inline Symbol Softmax(Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::kNull,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, y=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, y=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in src/operator/sequence_reverse.cc:L98
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Applies correlation to inputs.
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol();
}

/*!
 * \breif Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in src/operator/lrn.cc:L58
 * \param data Input data.
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in src/operator/pooling_v1.cc:L85
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(Symbol data,
                         Shape kernel,
                         Pooling_v1PoolType pool_type,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::kValid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in src/operator/fully_connected.cc:L74
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, y=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in src/operator/sequence_mask.cc:L112
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in src/operator/sequence_last.cc:L77
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceLast(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Generates 2D sampling grid for bilinear sampling.
 * \param data Input data to the function.
 * \param transform_type The type of transformation. For `affine`, input data should be
 *        an affine matrix of size (batch, 6). For `warp`, input data should be an
 * \param target_shape Specifies the output shape (H, W). This is required if
 *        transformation type is `affine`. If transformation type is `warp`, this
 * \return new symbol
 */
inline Symbol GridGenerator(Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::kNone,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::kNone) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *
 *
 *
 *        Defined in src/operator/activation.cc:L77
 * \param data Input array to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol();
}

/*!
 * \breif Applies dropout operation to input array.
 *
 *        - During training, each element of the input is set to zero with probability p.
 *        The whole array is rescaled by :math:`1/(1-p)` to keep the expected
 *        sum of the input unchanged.
 *
 *        - During testing, this operator does not change the input.
 *
 *        Example::
 *
 *        random.seed(998)
 *        input_array = array([[3., 0.5,  -0.5,  2., 7.],
 *        [2., -0.4,   7.,  3., 0.2]])
 *        a = symbol.Variable('a')
 *        dropout = symbol.Dropout(a, p = 0.2)
 *        executor = dropout.simple_bind(a = input_array.shape)
 *
 *        ## If training
 *        executor.forward(is_train = True, a = input_array)
 *        executor.outputs
 *        [[ 3.75   0.625 -0.     2.5    8.75 ]
 *        [ 2.5   -0.5    8.75   3.75   0.   ]]
 *
 *        ## If testing
 *        executor.forward(is_train = False, a = input_array)
 *        executor.outputs
 *        [[ 3.     0.5   -0.5    2.     7.   ]
 *        [ 2.    -0.4    7.     3.     0.2  ]]
 *
 *
 *        Defined in src/operator/dropout.cc:L62
 * \param data Input array to which dropout will be applied.
 * \param p Fraction of the input that gets dropped out during training time.
 * \return new symbol
 */
inline Symbol Dropout(Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
 *        width)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concatenating
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in src/operator/convolution.cc:L154
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temporary workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::kNone,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::kNone) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Computes and optimizes for squared loss during backward propagation.
 *        Just outputs ``data`` during forward propagation.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the squared loss estimated over :math:`n` samples is defined as
 *
 *        :math:`\text{SquaredLoss}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left(
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L51
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i -
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L72
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-x)}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in src/operator/regression_output.cc:L93
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in src/operator/make_loss.cc:L52
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::kNull) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in src/operator/roi_pooling.cc:L273
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol();
}

/*!
 * \breif Applies softmax activation to input. This is intended for internal layers.
 *
 *        .. note::
 *
 *        This operator has been deprecated, please use `softmax`.
 *
 *        If `mode` = ``instance``, this operator will compute a softmax for each
 *        This is the default mode.
 *
 *        If `mode` = ``channel``, this operator will compute a k-class softmax at each
 *        of each instance, where `k` = ``num_channel``. This mode can only be used when
 *        has at least 3 dimensions.
 *        This can be used for `fully convolutional network`, `image segmentation`, etc.
 *
 *        Example::
 *
 *        >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
 *        >>>                            [2., -.4, 7.,   3., 0.2]])
 *        >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
 *        >>> print softmax_act.asnumpy()
 *        [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03
 *        [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02
 *
 *
 *
 *        Defined in src/operator/softmax_activation.cc:L48
 * \param data Input array to activation function.
 * \param mode Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 * \return new symbol
 */
inline Symbol SoftmaxActivation(Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::kInstance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in src/operator/l2_normalization.cc:L74
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::kInstance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies a recurrent layer to input.
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol();
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs. This function assume rhs uses 0-based
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

} //namespace cpp
} //namespace mxnet
#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_
