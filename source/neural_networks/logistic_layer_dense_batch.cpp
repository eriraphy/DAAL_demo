/* file: logistic_layer_dense_batch.cpp */
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
!    C++ example of forward and backward logistic layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOGISTIC_LAYER_BATCH"></a>
 * \example logistic_layer_batch.cpp
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

int main()
{
    /* Retrieve the input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward logistic layer results using default method */
    logistic::forward::Batch<> logisticLayerForward;

    /* Set input objects for the forward logistic layer */
    logisticLayerForward.input.set(forward::data, tensorData);

    /* Compute forward logistic layer results */
    logisticLayerForward.compute();

    /* Print the results of the forward logistic layer */
    services::SharedPtr<logistic::forward::Result> forwardResult = logisticLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward logistic layer result (first 5 rows):", 5);

    /* Get the size of forward logistic layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward logistic layer results using default method */
    logistic::backward::Batch<> logisticLayerBackward;

    /* Set input objects for the backward logistic layer */
    logisticLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    logisticLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward logistic layer results */
    logisticLayerBackward.compute();

    /* Print the results of the backward logistic layer */
    services::SharedPtr<backward::Result> backwardResult = logisticLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward logistic layer result (first 5 rows):", 5);

    return 0;
}
