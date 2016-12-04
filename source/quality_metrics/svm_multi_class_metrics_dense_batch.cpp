/* file: svm_multi_class_metrics_dense_batch.cpp */
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
!    C++ example of multi-class support vector machine (SVM) quality metrics
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_QUALITY_METRIC_SET_BATCH"></a>
 * \example svm_multi_class_quality_metric_set_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::classifier::quality_metric;

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/svm_multi_class_train_dense.csv";

string testDatasetFileName      = "../data/batch/svm_multi_class_test_dense.csv";

const size_t nFeatures          = 20;
const size_t nClasses           = 5;

services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

/* Model object for the multi-class classifier algorithm */
services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;

/* Parameters for the multi-class classifier kernel function */
services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());

services::SharedPtr<multi_class_classifier::quality_metric_set::ResultCollection> qualityMetricSetResult;

NumericTablePtr predictedLabels;
NumericTablePtr groundTruthLabels;

void trainModel();
void testModel();
void testModelQuality();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

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
    groundTruthLabels = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, groundTruthLabels));

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

void testModelQuality()
{
    /* Retrieve predicted labels */
    predictedLabels = predictionResult->get(classifier::prediction::prediction);

    /* Create a quality metric set object to compute quality metrics of the multi-class classifier algorithm */
    multi_class_classifier::quality_metric_set::Batch qualityMetricSet(nClasses);

    services::SharedPtr<multiclass_confusion_matrix::Input> input =
            qualityMetricSet.getInputDataCollection()->getInput(multi_class_classifier::quality_metric_set::confusionMatrix);

    input->set(multiclass_confusion_matrix::predictedLabels,   predictedLabels);
    input->set(multiclass_confusion_matrix::groundTruthLabels, groundTruthLabels);

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
    services::SharedPtr<multiclass_confusion_matrix::Result> qualityMetricResult =
        qualityMetricSetResult->getResult(multi_class_classifier::quality_metric_set::confusionMatrix);
    printNumericTable(qualityMetricResult->get(multiclass_confusion_matrix::confusionMatrix), "Confusion matrix:");

    BlockDescriptor<> block;
    NumericTablePtr qualityMetricsTable = qualityMetricResult->get(multiclass_confusion_matrix::multiClassMetrics);
    qualityMetricsTable->getBlockOfRows(0, 1, readOnly, block);
    double *qualityMetricsData = block.getBlockPtr();
    std::cout << "Average accuracy: " << qualityMetricsData[multiclass_confusion_matrix::averageAccuracy    ] << std::endl;
    std::cout << "Error rate:       " << qualityMetricsData[multiclass_confusion_matrix::errorRate          ] << std::endl;
    std::cout << "Micro precision:  " << qualityMetricsData[multiclass_confusion_matrix::microPrecision     ] << std::endl;
    std::cout << "Micro recall:     " << qualityMetricsData[multiclass_confusion_matrix::microRecall        ] << std::endl;
    std::cout << "Micro F-score:    " << qualityMetricsData[multiclass_confusion_matrix::microFscore        ] << std::endl;
    std::cout << "Macro precision:  " << qualityMetricsData[multiclass_confusion_matrix::macroPrecision     ] << std::endl;
    std::cout << "Macro recall:     " << qualityMetricsData[multiclass_confusion_matrix::macroRecall        ] << std::endl;
    std::cout << "Macro F-score:    " << qualityMetricsData[multiclass_confusion_matrix::macroFscore        ] << std::endl;
    qualityMetricsTable->releaseBlockOfRows(block);
}
