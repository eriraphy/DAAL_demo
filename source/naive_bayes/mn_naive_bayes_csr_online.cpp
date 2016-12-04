/* file: mn_naive_bayes_csr_online.cpp */
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
!    C++ example of Naive Bayes classification in the online processing mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    compressed sparse rows (CSR)__format and then performs classification of
!    previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_ONLINE"></a>
 * \example multinomial_naive_bayes_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multinomial_naive_bayes;

/* Input data set parameters */
const string trainDatasetFileNames[4]     =
{
    "../data/online/naivebayes_train_csr_1.csv", "../data/online/naivebayes_train_csr_2.csv",
    "../data/online/naivebayes_train_csr_3.csv", "../data/online/naivebayes_train_csr_4.csv"
};
const string trainGroundTruthFileNames[4] =
{
    "../data/online/naivebayes_train_labels_1.csv", "../data/online/naivebayes_train_labels_2.csv",
    "../data/online/naivebayes_train_labels_3.csv", "../data/online/naivebayes_train_labels_4.csv"
};

string testDatasetFileName      = "../data/online/naivebayes_test_csr.csv";
string testGroundTruthFileName  = "../data/online/naivebayes_test_labels.csv";

const size_t nTrainVectorsInBlock = 8000;
const size_t nTestObservations    = 2000;
const size_t nClasses             = 20;
const size_t nBlocks              = 4;

services::SharedPtr<training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
services::SharedPtr<CSRNumericTable> trainData[nBlocks];
services::SharedPtr<CSRNumericTable> testData;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &trainDatasetFileNames, &trainGroundTruthFileNames, &testDatasetFileName,
                   &testGroundTruthFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Create an algorithm object to train the Naive Bayes model */
    training::Online<double, training::fastCSR> algorithm(nClasses);

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Read trainDatasetFileNames and create a numeric table to store the input data */
        trainData[i] = services::SharedPtr<CSRNumericTable>(createSparseTable<double>(trainDatasetFileNames[i]));
        FileDataSource<CSVFeatureManager> trainLabelsSource(trainGroundTruthFileNames[i],
                                                        DataSource::doAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);
        trainLabelsSource.loadDataBlock(nTrainVectorsInBlock);
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(classifier::training::data,   trainData[i]);
        algorithm.input.set(classifier::training::labels, trainLabelsSource.getNumericTable());

        /* Build the Naive Bayes model */
        algorithm.compute();
    }
    /* Finalize the Naive Bayes model */
    algorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Read testDatasetFileName and create a numeric table to store the input data */
    testData = services::SharedPtr<CSRNumericTable>(createSparseTable<double>(testDatasetFileName));

    /* Create an algorithm object to predict Naive Bayes values */
    prediction::Batch<double, prediction::fastCSR> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data,  testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict Naive Bayes values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    FileDataSource<CSVFeatureManager> testGroundTruth(testGroundTruthFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testGroundTruth.loadDataBlock(nTestObservations);

    printNumericTables<int, int>(testGroundTruth.getNumericTable().get(),
                                 predictionResult->get(classifier::prediction::prediction).get(),
                                 "Ground truth", "Classification results",
                                 "NaiveBayes classification results (first 20 observations):", 20);
}
