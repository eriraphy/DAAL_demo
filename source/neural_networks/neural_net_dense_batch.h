/* file: neural_net_dense_batch.h */
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

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

enum LayerIndex
{
    fc1 = 0,
    fc2 = 1,
    sm1 = 2
};

training::Topology configureNet()
{
    /* Create layers of the neural network */
    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer1(new fullyconnected::Batch<>(5));

    fullyConnectedLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer1->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0, 0.5));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer2(new fullyconnected::Batch<>(2));

    fullyConnectedLayer2->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(0.5, 1));

    fullyConnectedLayer2->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0.5, 1));

    /* Create softmax layer and initialize layer parameters */
    SharedPtr<loss::softmax_cross::Batch<> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<>());

    /* Create topology of the neural network */
    training::Topology topology;

    /* Add layers to the topology of the neural network */
    topology.push_back(LayerDescriptor(fc1, fullyConnectedLayer1, NextLayers(fc2)));
    topology.push_back(LayerDescriptor(fc2, fullyConnectedLayer2, NextLayers(sm1)));
    topology.push_back(LayerDescriptor(sm1, softmaxCrossEntropyLayer, NextLayers()));

    return topology;
}
