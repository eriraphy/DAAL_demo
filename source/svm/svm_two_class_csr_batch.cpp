/* file: svm_two_class_csr_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_CSR_BATCH"></a>
 * \example svm_two_class_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/svm_two_class_train_csr.csv";
string trainLabelsFileName      = "../data/batch/svm_two_class_train_labels.csv";

string testDatasetFileName      = "../data/batch/svm_two_class_test_csr.csv";
string testLabelsFileName       = "../data/batch/svm_two_class_test_labels.csv";

/* Parameters for the SVM kernel function */
services::SharedPtr<kernel_function::KernelIface> kernel(
    new kernel_function::linear::Batch<double, kernel_function::linear::fastCSR>());

/* Model object for the SVM algorithm */
services::SharedPtr<svm::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &trainDatasetFileName, &trainLabelsFileName, &testDatasetFileName, &testLabelsFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsDataSource(trainLabelsFileName,
                                                            DataSource::doAllocateNumericTable,
                                                            DataSource::doDictionaryFromContext);

    /* Create numeric table for training data */
    services::SharedPtr<CSRNumericTable> trainData(createSparseTable<double>(trainDatasetFileName));

    /* Retrieve the data from the input file */
    trainLabelsDataSource.loadDataBlock();

    /* Create an algorithm object to train the SVM model */
    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;
    algorithm.parameter.cacheSize = 40000000;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainLabelsDataSource.getNumericTable());

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Create Numeric Tables for testing data */
    NumericTablePtr testData(createSparseTable<double>(testDatasetFileName));

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testLabelsDataSource(testLabelsFileName,
                                                           DataSource::doAllocateNumericTable,
                                                           DataSource::doDictionaryFromContext);
    /* Retrieve the data from input file */
    testLabelsDataSource.loadDataBlock();
    NumericTablePtr testGroundTruth = testLabelsDataSource.getNumericTable();

    printNumericTables<int, double>(testGroundTruth,
                                    predictionResult->get(classifier::prediction::prediction),
                                    "Ground truth\t", "Classification results",
                                    "SVN classification results (first 20 observations):", 20);
}
