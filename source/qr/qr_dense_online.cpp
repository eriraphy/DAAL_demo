/* file: qr_dense_online.cpp */
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
!    C++ example of computing QR decomposition in the online processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_ONLINE"></a>
 * \example qr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/online/qr.csv";
const size_t nRowsInBlock = 4000;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute QR decomposition in the online processing mode */
    qr::Online<> algorithm;

    while(dataSource.loadDataBlock(nRowsInBlock) == nRowsInBlock)
    {
        algorithm.input.set( qr::data, dataSource.getNumericTable() );

        /* Compute QR decomposition */
        algorithm.compute();
    }

    /* Finalize computations and retrieve the results */
    algorithm.finalizeCompute();

    services::SharedPtr<qr::Result> res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(qr::matrixQ), "Orthogonal matrix Q:", 10);
    printNumericTable(res->get(qr::matrixR), "Triangular matrix R:");

    return 0;
}
