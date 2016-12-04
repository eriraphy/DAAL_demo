/* file: conv2d_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
!  Content:
!    C++ example of forward and backward two-dimensional convolution layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CONVOLUTION2D_LAYER_BATCH"></a>
 * \example convolution2d_layer_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set name */
string datasetFileName = "../data/batch/layer.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Create collection of dimension sizes of the input data tensor */
    Collection<size_t> inDims;
    inDims.push_back(2);
    inDims.push_back(1);
    inDims.push_back(16);
    inDims.push_back(16);
    TensorPtr tensorData = TensorPtr(new HomogenTensor<float>(inDims, Tensor::doAllocate, 1.0f));

    /* Create an algorithm to compute forward two-dimensional convolution layer results using default method */
    convolution2d::forward::Batch<> convolution2dLayerForward;
    convolution2dLayerForward.input.set(forward::data, tensorData);

    /* Compute forward two-dimensional convolution layer results */
    convolution2dLayerForward.compute();

    /* Get the computed forward two-dimensional convolution layer results */
    services::SharedPtr<convolution2d::forward::Result> forwardResult = convolution2dLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Two-dimensional convolution layer result (first 5 rows):", 5, 15);
    printTensor(forwardResult->get(convolution2d::auxWeights), "Two-dimensional convolution layer weights (first 5 rows):", 5, 15);

    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    /* Create input gradient tensor for backward two-dimensional convolution layer */
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward two-dimensional convolution layer results using default method */
    convolution2d::backward::Batch<> convolution2dLayerBackward;
    convolution2dLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    convolution2dLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional convolution layer results */
    convolution2dLayerBackward.compute();

    /* Get the computed backward two-dimensional convolution layer results */
    services::SharedPtr<backward::Result> backwardResult = convolution2dLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient),
                "Two-dimensional convolution layer backpropagation gradient result (first 5 rows):", 5, 15);
    printTensor(backwardResult->get(backward::weightDerivatives),
                "Two-dimensional convolution layer backpropagation weightDerivative result (first 5 rows):", 5, 15);
    printTensor(backwardResult->get(backward::biasDerivatives),
                "Two-dimensional convolution layer backpropagation biasDerivative result (first 5 rows):", 5, 15);

    return 0;
}
