/* file: neural_net_dense_batch.cpp */
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
!    C++ example of neural network training and scoring
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_BATCH"></a>
 * \example neural_network_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_net_dense_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string trainDatasetFile     = "../data/batch/neural_network_train.csv";
string trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

services::SharedPtr<prediction::Model> predictionModel;
services::SharedPtr<prediction::Result> predictionResult;

void trainModel();
void testModel();
void printResults();

int main()
{
    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Read training data set from a .csv file and create a tensor to store input data */
    TensorPtr trainingData = readTensorFromCSV(trainDatasetFile);
    TensorPtr trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile);

    /* Create an algorithm to train neural network */
    training::Batch<> net;

    /* Set the batch size for the neural network training */
    net.parameter.batchSize = 10;

    /* Configure the neural network */
    training::Topology topology = configureNet();
    net.initialize(trainingData->getDimensions(), topology);

    /* Pass a training data set and dependent values to the algorithm */
    net.input.set(training::data, trainingData);
    net.input.set(training::groundTruth, trainingGroundTruth);

    /* Create stochastic gradient descent (SGD) optimization solver algorithm */
    SharedPtr<optimization_solver::sgd::Batch<float> > sgdAlgorithm(new optimization_solver::sgd::Batch<float>());

    /* Set learning rate for the optimization solver used in the neural network */

    float learningRate = 0.001f;
    sgdAlgorithm->parameter.learningRateSequence = NumericTablePtr(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));

    /* Set the optimization solver for the neural network training */
    net.parameter.optimizationSolver = sgdAlgorithm;

    /* Run the neural network training */
    net.compute();

    /* Retrieve training and prediction models of the neural network */
    SharedPtr<training::Model> trainingModel = net.getResult()->get(training::model);
    predictionModel = trainingModel->getPredictionModel<float>();
}

void testModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    TensorPtr predictionData = readTensorFromCSV(testDatasetFile);

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
