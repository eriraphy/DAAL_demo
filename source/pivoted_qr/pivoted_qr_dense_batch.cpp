/* file: pivoted_qr_dense_batch.cpp */
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
!    C++ example of computing pivoted QR decomposition
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PIVOTED_QR_BATCH"></a>
 * \example pivoted_qr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/qr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute pivoted QR decomposition */
    pivoted_qr::Batch<> algorithm;

    algorithm.input.set(pivoted_qr::data, dataSource.getNumericTable());

    /* Compute pivoted QR decomposition */
    algorithm.compute();

    services::SharedPtr<pivoted_qr::Result> res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(pivoted_qr::matrixQ), "Orthogonal matrix Q:", 10);
    printNumericTable(res->get(pivoted_qr::matrixR), "Triangular matrix R:");
    printNumericTable(res->get(pivoted_qr::permutationMatrix), "Permutation matrix P:");

    return 0;
}
