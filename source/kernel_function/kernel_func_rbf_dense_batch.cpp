/* file: kernel_func_rbf_dense_batch.cpp */
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
!    C++ example of computing a radial basis function (RBF) kernel
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KERNEL_FUNCTION_RBF_DENSE_BATCH"></a>
 * \example kernel_function_rbf_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string leftDatasetFileName  = "../data/batch/kernel_function.csv";
string rightDatasetFileName = "../data/batch/kernel_function.csv";

/* Kernel algorithm parameters */
const double sigma = 1.0;       /* RBF kernel coefficient */

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &leftDatasetFileName);
    checkArguments(argc, argv, 1, &rightDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> leftDataSource(leftDatasetFileName, DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    FileDataSource<CSVFeatureManager> rightDataSource(rightDatasetFileName, DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    leftDataSource.loadDataBlock();
    rightDataSource.loadDataBlock();

    /* Create algorithm objects for the kernel algorithm using the default method */
    kernel_function::rbf::Batch<> algorithm;

    /* Set the kernel algorithm parameter */
    algorithm.parameter.sigma = sigma;
    algorithm.parameter.computationMode = kernel_function::matrixMatrix;

    /* Set an input data table for the algorithm */
    algorithm.input.set(kernel_function::X, leftDataSource.getNumericTable());
    algorithm.input.set(kernel_function::Y, rightDataSource.getNumericTable());

    /* Compute the RBF kernel */
    algorithm.compute();

    /* Get the computed results */
    services::SharedPtr<kernel_function::Result> result = algorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(kernel_function::values), "Values");

    return 0;
}
