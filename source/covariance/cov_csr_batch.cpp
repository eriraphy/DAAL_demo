/* file: cov_csr_batch.cpp */
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
!    C++ example of variance-covariance matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_CSR_BATCH"></a>
 * \example covariance_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float   dataFPType;         /* Input data floating-point type */
typedef double  algorithmFPType;    /* Algorithm floating-point type */

/* Input data set parameters
   Input matrix is stored in the compressed sparse row format with one-based indexing
 */
const string datasetFileName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    services::SharedPtr<CSRNumericTable> dataTable(createSparseTable<dataFPType>(datasetFileName));

    /* Create an algorithm to compute variance-covariance matrix using the default method */
    covariance::Batch<algorithmFPType, covariance::fastCSR> algorithm;
    algorithm.input.set(covariance::data, dataTable);

    /* Compute a variance-covariance matrix */
    algorithm.compute();

    /* Get the computed variance-covariance matrix */
    services::SharedPtr<covariance::Result> res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance), "Covariance matrix (upper left square 10*10) :", 10, 10);
    printNumericTable(res->get(covariance::mean),       "Mean vector:", 1, 10);

    return 0;
}
