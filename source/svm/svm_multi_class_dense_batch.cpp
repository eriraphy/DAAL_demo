/* file: svm_multi_class_dense_batch.cpp */
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
!    C++ example of multi-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_DENSE_BATCH"></a>
 * \example svm_multi_class_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName     = "C:/Users/Raphy/Desktop/C/data/batch/svm_multi_class_train_dense.csv";

string testDatasetFileName      = "C:/Users/Raphy/Desktop/C/data/batch/svm_multi_class_test_dense.csv";

const size_t nFeatures          = 20;
const size_t nClasses           = 5;

services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the multi-class SVM model */
    multi_class_classifier::training::Batch<> algorithm;

    algorithm.parameter.nClasses = nClasses;
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the multi-class SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict multi-class SVM values */
    multi_class_classifier::prediction::Batch<> algorithm;

    algorithm.parameter.nClasses = nClasses;
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict multi-class SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth", "Classification results",
                                 "Multi-class SVM classification sample program results (first 20 observations):", 20);
}
