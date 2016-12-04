/* file: prelu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward parametric rectified linear unit (prelu) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PRELU_LAYER_BATCH"></a>
 * \example prelu_layer_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set parameters */
string datasetName = "../data/batch/layer.csv";
string weightsName = "../data/batch/layer.csv";

size_t dataDimension = 0;
size_t weightsDimension = 2;

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);
    TensorPtr tensorWeights = readTensorFromCSV(weightsName);

    /* Create an algorithm to compute forward prelu layer results using default method */
    prelu::forward::Batch<> forwardPreluLayer;
    forwardPreluLayer.parameter.dataDimension = dataDimension;
    forwardPreluLayer.parameter.weightsDimension = weightsDimension;

    /* Set input objects for the forward prelu layer */
    forwardPreluLayer.input.set(forward::data, tensorData);
    forwardPreluLayer.input.set(forward::weights, tensorWeights);

    /* Compute forward prelu layer results */
    forwardPreluLayer.compute();

    /* Print the results of the forward prelu layer */
    services::SharedPtr<prelu::forward::Result> forwardResult = forwardPreluLayer.getResult();
    printTensor(forwardResult->get(forward::value), "Forward prelu layer result (first 5 rows):", 5);

    /* Get the size of forward prelu layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward prelu layer results using default method */
    prelu::backward::Batch<> backwardPreluLayer;
    backwardPreluLayer.parameter.dataDimension = dataDimension;
    backwardPreluLayer.parameter.weightsDimension = weightsDimension;

    /* Set input objects for the backward prelu layer */
    backwardPreluLayer.input.set(backward::inputGradient, tensorDataBack);
    backwardPreluLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward prelu layer results */
    backwardPreluLayer.compute();

    /* Print the results of the backward prelu layer */
    services::SharedPtr<backward::Result> backwardResult = backwardPreluLayer.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward prelu layer result (first 5 rows):", 5);
    printTensor(backwardResult->get(backward::weightDerivatives), "Weights derivative (first 5 rows):", 5);

    return 0;
}
