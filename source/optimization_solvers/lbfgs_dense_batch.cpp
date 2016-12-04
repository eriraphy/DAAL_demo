/* file: lbfgs_dense_batch.cpp */
/*******************************************************************************
!  Copyright(C) 2014-2016 Intel Corporation. All Rights Reserved.
!
!  The source code, information  and  material ("Material") contained herein is
!  owned  by Intel Corporation or its suppliers or licensors, and title to such
!  Material remains  with Intel Corporation  or its suppliers or licensors. The
!  Material  contains proprietary information  of  Intel or  its  suppliers and
!  licensors. The  Material is protected by worldwide copyright laws and treaty
!  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!  modified, published, uploaded, posted, transmitted, distributed or disclosed
!  in any way  without Intel's  prior  express written  permission. No  license
!  under  any patent, copyright  or  other intellectual property rights  in the
!  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!  intellectual  property  rights must  be express  and  approved  by  Intel in
!  writing.
!
!  *Third Party trademarks are the property of their respective owners.
!
!  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!  this  notice or  any other notice embedded  in Materials by Intel or Intel's
!  suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!    C++ example of the limited memory Broyden-Fletcher-Goldfarb-Shanno
!    algorithm
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LBFGS_BATCH"></a>
 * \example lbfgs_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string datasetFileName = "../data/batch/lbfgs.csv";

const size_t nFeatures   = 10;
const size_t nIterations = 1000;
const double stepLength  = 1.0e-4;

double initialPoint[nFeatures + 1]  = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
double expectedPoint[nFeatures + 1] = { 11,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for input data and dependent variables */
    NumericTablePtr data(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from input file */
    dataSource.loadDataBlock(mergedData.get());

    services::SharedPtr<optimization_solver::mse::Batch<> > mseObjectiveFunction(
        new optimization_solver::mse::Batch<>(data->getNumberOfRows()));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables, dependentVariables);

    /* Create objects to compute LBFGS result using the default method */
    optimization_solver::lbfgs::Batch<> algorithm(mseObjectiveFunction);
    algorithm.parameter.nIterations = nIterations;
    algorithm.parameter.stepLengthSequence =
        NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTableIface::doAllocate, stepLength));

    /* Set input objects for LBFGS algorithm */
    algorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                        NumericTablePtr(new HomogenNumericTable<>(initialPoint, nFeatures + 1, 1)));

    /* Compute LBFGS result */
    algorithm.compute();

    NumericTablePtr expectedCoefficients =
        NumericTablePtr(new HomogenNumericTable<>(expectedPoint, nFeatures + 1, 1));

    /* Print computed LBFGS results */
    printNumericTable(expectedCoefficients,
                      "Expected coefficients:");
    printNumericTable(algorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
                      "Resulting coefficients:");
    printNumericTable(algorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
                      "Number of iterations performed:");
    return 0;
}
