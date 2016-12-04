/* file: neural_net_predict_dense_batch.h */
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

prediction::Topology configureNet()
{
    /* Create layers of the neural network */
    /* Create first fully-connected layer */
    SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer1(new fullyconnected::forward::Batch<>(5));

    /* Create second fully-connected layer */
    SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer2(new fullyconnected::forward::Batch<>(2));

    /* Create softmax layer */
    SharedPtr<softmax::forward::Batch<> > softmaxLayer(new softmax::forward::Batch<>());

    /* Create topology of the neural network */
    prediction::Topology topology;

    /* Add layers to the topology of the neural network */
    topology.push_back(forward::LayerDescriptor(fc1, fullyConnectedLayer1, NextLayers(fc2)));
    topology.push_back(forward::LayerDescriptor(fc2, fullyConnectedLayer2, NextLayers(sm1)));
    topology.push_back(forward::LayerDescriptor(sm1, softmaxLayer, NextLayers()));

    return topology;
}
