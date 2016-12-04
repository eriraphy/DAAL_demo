/* file: sgd_mini_dense_batch.cpp */
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
!    C++ example of the Stochastic gradient descent algorithm
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SGD_MINI_BATCH"></a>
 * \example sgd_mini_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string datasetFileName = "../data/batch/mse.csv";

const size_t nFeatures = 3;
const double accuracyThreshold = 0.0000001;
const size_t nIterations = 1000;
const size_t batchSize = 4;
const double learningRate = 0.5;
double initialPoint[nFeatures + 1] = {8, 2, 1, 4};

int main(int argc, char *argv[])
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
            DataSource::notAllocateNumericTable,
            DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    NumericTablePtr data(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());

    size_t nVectors = data->getNumberOfRows();

    services::SharedPtr<optimization_solver::mse::Batch<double> > mseObjectiveFunction(new optimization_solver::mse::Batch<double>(nVectors));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables, dependentVariables);

    /* Create objects to compute the Stochastic gradient descent result using the mini-batch method */
    optimization_solver::sgd::Batch<double, optimization_solver::sgd::miniBatch> sgdMiniBatchAlgorithm(mseObjectiveFunction);

    /* Set input objects for the the Stochastic gradient descent algorithm */
    sgdMiniBatchAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                                    NumericTablePtr(new HomogenNumericTable<double>(initialPoint, nFeatures + 1, 1)));
    sgdMiniBatchAlgorithm.parameter.learningRateSequence =
        NumericTablePtr(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));
    sgdMiniBatchAlgorithm.parameter.nIterations = nIterations;
    sgdMiniBatchAlgorithm.parameter.batchSize = batchSize;
    sgdMiniBatchAlgorithm.parameter.accuracyThreshold = accuracyThreshold;

    /* Compute the Stochastic gradient descent result */
    sgdMiniBatchAlgorithm.compute();

    /* Print computed the Stochastic gradient descent result */
    printNumericTable(sgdMiniBatchAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum), "Minimum");
    printNumericTable(sgdMiniBatchAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "Number of iterations performed:");

    return 0;
}
