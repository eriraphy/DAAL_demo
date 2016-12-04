/* file: split_layer_dense_batch.cpp */
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
!    C++ example of forward and backward split layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SPLIT_LAYER_BATCH"></a>
 * \example split_layer_batch.cpp
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
const size_t nOutputs = 3;
const size_t nInputs  = 3;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward split layer results using default method */
    split::forward::Batch<> splitLayerForward;

    /* Set parameters for the forward split layer */
    splitLayerForward.parameter.nOutputs = nOutputs;
    splitLayerForward.parameter.nInputs = nInputs;

    /* Set input objects for the forward split layer */
    splitLayerForward.input.set(forward::data, tensorData);

    printTensor(tensorData, "Split layer input (first 5 rows):", 5);

    /* Compute forward split layer results */
    splitLayerForward.compute();

    /* Print the results of the forward split layer */
    services::SharedPtr<split::forward::Result> forwardResult = splitLayerForward.getResult();

    for(size_t i = 0; i < nOutputs; i++)
    {
        printTensor(forwardResult->get(split::forward::valueCollection, i), "Forward split layer result (first 5 rows):", 5);
    }

    /* Create an algorithm to compute backward split layer results using default method */
    split::backward::Batch<> splitLayerBackward;

    /* Set parameters for the backward split layer */
    splitLayerBackward.parameter.nOutputs = nOutputs;
    splitLayerBackward.parameter.nInputs = nInputs;

    /* Set input objects for the backward split layer */
    splitLayerBackward.input.set(split::backward::inputGradientCollection, forwardResult->get(split::forward::valueCollection));

    /* Compute backward split layer results */
    splitLayerBackward.compute();

    /* Print the results of the backward split layer */
    services::SharedPtr<backward::Result> backwardResult = splitLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward split layer result (first 5 rows):", 5);

    return 0;
}
