/* file: neural_net_predict_dense_batch.cpp */
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
!    C++ example of neural network scoring
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_PREDICTION_BATCH"></a>
 * \example neural_network_prediction_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_net_predict_dense_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

/* Weights and biases obtained on the training stage */
string fc1WeightsFile = "../data/batch/fc1_weights.csv";
string fc1BiasesFile  = "../data/batch/fc1_biases.csv";
string fc2WeightsFile = "../data/batch/fc2_weights.csv";
string fc2BiasesFile  = "../data/batch/fc2_biases.csv";

TensorPtr predictionData;
services::SharedPtr<prediction::Model> predictionModel;
services::SharedPtr<prediction::Result> predictionResult;

void createModel();
void testModel();
void printResults();

int main()
{
    createModel();

    testModel();

    printResults();

    return 0;
}

void createModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    predictionData = readTensorFromCSV(testDatasetFile);

    /* Configure the neural network */
    prediction::Topology topology = configureNet();

    /* Create prediction model of the neural network */
    predictionModel = services::SharedPtr<prediction::Model>(new prediction::Model(topology));

    /* Read 1st fully-connected layer weights and biases from CSV file */
    /* 1st fully-connected layer weights are a 2D tensor of size 5 x 20 */
    TensorPtr fc1Weights = readTensorFromCSV(fc1WeightsFile);
    /* 1st fully-connected layer biases are a 1D tensor of size 5 */
    TensorPtr fc1Biases = readTensorFromCSV(fc1BiasesFile);

    /* Set weights and biases of the 1st fully-connected layer */
    forward::Input *fc1Input = predictionModel->getLayer(fc1)->getLayerInput();
    fc1Input->set(forward::weights, fc1Weights);
    fc1Input->set(forward::biases, fc1Biases);

    /* Read 2nd fully-connected layer weights and biases from CSV file */
    /* 2nd fully-connected layer weights are a 2D tensor of size 2 x 5 */
    TensorPtr fc2Weights = readTensorFromCSV(fc2WeightsFile);
    /* 2nd fully-connected layer biases are a 1D tensor of size 2 */
    TensorPtr fc2Biases = readTensorFromCSV(fc2BiasesFile);

    /* Set weights and biases of the 2nd fully-connected layer */
    forward::Input *fc2Input = predictionModel->getLayer(fc2)->getLayerInput();
    fc2Input->set(forward::weights, fc2Weights);
    fc2Input->set(forward::biases, fc2Biases);

    /* Allocate memory for prediction model of the neural network */
    predictionModel->allocate<float>(predictionData->getDimensions());
}

void testModel()
{
    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set input objects for the prediction neural network */
    net.input.set(prediction::model, predictionModel);
    net.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    net.compute();

    /* Print results of the neural network prediction */
    predictionResult = net.getResult();
}

void printResults()
{
    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
}
