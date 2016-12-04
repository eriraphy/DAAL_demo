/* file: ridge_reg_norm_eq_distr.cpp */
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
!    C++ example of ridge regression in the distributed processing mode.
!
!    The program trains the multiple ridge regression model on a training
!    datasetFileName with the normal equations method and computes regression
!    for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-RIDGE_REGRESSION_NORM_EQ_DISTRIBUTED"></a>
 * \example ridge_regression_norm_eq_distributed.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::ridge_regression;

const string trainDatasetFileNames[] =
{
    "../data/distributed/linear_regression_train_1.csv", "../data/distributed/linear_regression_train_2.csv",
    "../data/distributed/linear_regression_train_3.csv", "../data/distributed/linear_regression_train_4.csv"
};

string testDatasetFileName    = "../data/distributed/linear_regression_test.csv";

const size_t nBlocks              = 4;

const size_t nFeatures            = 10;
const size_t nDependentVariables  = 2;

void trainModel();
void testModel();

services::SharedPtr<training::Result> trainingResult;
services::SharedPtr<prediction::Result> predictionResult;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 5, &testDatasetFileName,
                   &trainDatasetFileNames[0], &trainDatasetFileNames[1],
                   &trainDatasetFileNames[2], &trainDatasetFileNames[3]);

    trainModel();
    testModel();

    return 0;
}

void trainModel()
{
    /* Create an algorithm object to build the final ridge regression model on the master node */
    training::Distributed<step2Master> masterAlgorithm;

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileNames[i],
                                                          DataSource::notAllocateNumericTable,
                                                          DataSource::doDictionaryFromContext);

        /* Create Numeric Tables for training data and variables */
        NumericTablePtr trainData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
        NumericTablePtr trainDependentVariables(new HomogenNumericTable<double>(nDependentVariables, 0, NumericTable::notAllocate));
        NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVariables));

        /* Retrieve the data from input file */
        trainDataSource.loadDataBlock(mergedData.get());

        /* Create an algorithm object to train the ridge regression model based on the local-node data */
        training::Distributed<step1Local> localAlgorithm;

        /* Pass a training data set and dependent values to the algorithm */
        localAlgorithm.input.set(training::data, trainData);
        localAlgorithm.input.set(training::dependentVariables, trainDependentVariables);

        /* Train the ridge regression model on the local-node data */
        localAlgorithm.compute();

        /* Set the local ridge regression model as input for the master-node algorithm */
        masterAlgorithm.input.add(training::partialModels, localAlgorithm.getPartialResult());
    }

    /* Merge and finalize the ridge regression model on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = masterAlgorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(), "Ridge Regression coefficients:");
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr testGroundTruth(new HomogenNumericTable<double>(nDependentVariables, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Load the data from the data file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict values of ridge regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of ridge regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction), "Ridge Regression prediction results: (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}
