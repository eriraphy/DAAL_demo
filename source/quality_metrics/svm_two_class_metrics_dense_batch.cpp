/* file: svm_two_class_metrics_dense_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) quality metrics
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_QUALITY_METRIC_SET_BATCH"></a>
 * \example svm_two_class_quality_metric_set_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::classifier::quality_metric;

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/svm_two_class_train_dense.csv";

string testDatasetFileName      = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures          = 20;

/* Parameters for the SVM kernel function */
services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());

/* Model object for the SVM algorithm */
services::SharedPtr<svm::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
services::SharedPtr<svm::quality_metric_set::ResultCollection> qualityMetricSetResult;

NumericTablePtr predictedLabels;
NumericTablePtr groundTruthLabels;

void trainModel();
void testModel();
void testModelQuality();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();

    testModel();

    testModelQuality();

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

    /* Create an algorithm object to train the SVM model */
    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;
    algorithm.parameter.cacheSize = 40000000;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    groundTruthLabels = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, groundTruthLabels));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

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

void testModelQuality()
{
    /* Retrieve predicted labels */
    predictedLabels = predictionResult->get(classifier::prediction::prediction);

    /* Create a quality metric set object to compute quality metrics of the SVM algorithm */
    svm::quality_metric_set::Batch qualityMetricSet;

    services::SharedPtr<binary_confusion_matrix::Input> input =
            qualityMetricSet.getInputDataCollection()->getInput(svm::quality_metric_set::confusionMatrix);
    input->set(binary_confusion_matrix::predictedLabels,   predictedLabels);
    input->set(binary_confusion_matrix::groundTruthLabels, groundTruthLabels);

    /* Compute quality metrics */
    qualityMetricSet.compute();

    /* Retrieve the quality metrics */
    qualityMetricSetResult = qualityMetricSet.getResultCollection();
}

void printResults()
{
    /* Print the classification results */
    printNumericTables<int, double>(groundTruthLabels.get(), predictedLabels.get(),
                                    "Ground truth", "Classification results",
                                    "SVN classification results (first 20 observations):", 20);

    /* Print the quality metrics */
    services::SharedPtr<binary_confusion_matrix::Result> qualityMetricResult =
        qualityMetricSetResult->getResult(svm::quality_metric_set::confusionMatrix);
    printNumericTable(qualityMetricResult->get(binary_confusion_matrix::confusionMatrix), "Confusion matrix:");

    BlockDescriptor<> block;
    NumericTablePtr qualityMetricsTable = qualityMetricResult->get(binary_confusion_matrix::binaryMetrics);
    qualityMetricsTable->getBlockOfRows(0, 1, readOnly, block);
    double *qualityMetricsData = block.getBlockPtr();
    std::cout << "Accuracy:      " << qualityMetricsData[binary_confusion_matrix::accuracy   ] << std::endl;
    std::cout << "Precision:     " << qualityMetricsData[binary_confusion_matrix::precision  ] << std::endl;
    std::cout << "Recall:        " << qualityMetricsData[binary_confusion_matrix::recall     ] << std::endl;
    std::cout << "F-score:       " << qualityMetricsData[binary_confusion_matrix::fscore     ] << std::endl;
    std::cout << "Specificity:   " << qualityMetricsData[binary_confusion_matrix::specificity] << std::endl;
    std::cout << "AUC:           " << qualityMetricsData[binary_confusion_matrix::AUC        ] << std::endl;
    qualityMetricsTable->releaseBlockOfRows(block);
}
