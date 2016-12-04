/* file: mse_dense_batch.cpp */
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
!    C++ example of the mean squared error objective function
!******************************************************************************/


/**
 * <a name="DAAL-EXAMPLE-CPP-MSE_BATCH"></a>
 * \example mse_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string datasetFileName = "../data/batch/mse.csv";
const size_t nFeatures = 3;

double argumentValue[nFeatures + 1] = { -1, 0.1, 0.15, -0.5};

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

    /* Create the MSE objective function objects to compute the MSE objective function result using the default method */
    optimization_solver::mse::Batch<double> mseObjectiveFunction(nVectors);

    /* Set input objects for the MSE objective function */
    mseObjectiveFunction.input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction.input.set(optimization_solver::mse::dependentVariables, dependentVariables);
    mseObjectiveFunction.input.set(optimization_solver::mse::argument,
                                   NumericTablePtr(new HomogenNumericTable<double>(argumentValue, nFeatures + 1, 1)));
    mseObjectiveFunction.parameter.resultsToCompute =
        optimization_solver::objective_function::gradient |
        optimization_solver::objective_function::value |
        optimization_solver::objective_function::hessian;

    /* Compute the MSE objective function result */
    mseObjectiveFunction.compute();

    /* Print computed the MSE objective function result */
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::resultCollection,
                      optimization_solver::objective_function::valueIdx), "Value");
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::resultCollection,
                      optimization_solver::objective_function::gradientIdx), "Gradient");
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::resultCollection,
                      optimization_solver::objective_function::hessianIdx), "Hessian");

    return 0;
}
