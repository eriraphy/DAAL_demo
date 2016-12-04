/* file: max_pool3d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward three-dimensional maximum pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MAXIMUM_POOLING3D_LAYER_BATCH"></a>
 * \example maximum_pooling3d_layer_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

static const size_t nDim = 3;
static const size_t dims[] = {3, 2, 4};
static float dataArray[3][2][4] = {{{ 1,  2,  3,  4},
                                    { 5,  6,  7,  8}},
                                                    {{ 9, 10, 11, 12},
                                                    {13, 14, 15, 16}},
                                                                    {{17, 18, 19, 20},
                                                                     {21, 22, 23, 24}}};

int main(int argc, char *argv[])
{
    TensorPtr dataTensor(new HomogenTensor<float>(nDim, dims, (float *)dataArray));

    printTensor3d(dataTensor, "Forward maximum pooling layer input:");

    /* Create an algorithm to compute forward pooling layer results using maximum method */
    maximum_pooling3d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, dataTensor);

    /* Compute forward pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward pooling layer results */
    services::SharedPtr<maximum_pooling3d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor3d(forwardResult->get(forward::value),
        "Forward maximum pooling layer result:");
    printTensor3d(forwardResult->get(maximum_pooling3d::auxSelectedIndices),
        "Forward maximum pooling layer selected indices:");

    /* Create an algorithm to compute backward pooling layer results using maximum method */
    maximum_pooling3d::backward::Batch<> backwardLayer(nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward pooling layer results */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor3d(backwardResult->get(backward::gradient),
        "Backward maximum pooling layer result:");

    return 0;
}
