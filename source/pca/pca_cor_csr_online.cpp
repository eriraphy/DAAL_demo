/* file: pca_cor_csr_online.cpp */
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
!    C++ example of principal component analysis (PCA) using the correlation
!    method in the online processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_ONLINE"></a>
 * \example pca_correlation_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float  dataFPType;          /* Data floating-point type */
typedef double algorithmFPType;     /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;
const string datasetFileNames[] =
{
    "../data/online/covcormoments_csr_1.csv",
    "../data/online/covcormoments_csr_2.csv",
    "../data/online/covcormoments_csr_3.csv",
    "../data/online/covcormoments_csr_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Online<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance = services::SharedPtr<covariance::Online<algorithmFPType, covariance::fastCSR> >
                                     (new covariance::Online<algorithmFPType, covariance::fastCSR>());

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Read data from a file and create a numeric table to store input data */
        services::SharedPtr<CSRNumericTable> dataTable(createSparseTable<dataFPType>(datasetFileNames[i]));

        /* Set input objects for the algorithm */
        algorithm.input.set(pca::data, services::SharedPtr<CSRNumericTable>(dataTable));

        /* Update PCA decomposition */
        algorithm.compute();
    }

    /* Finalize computations */
    algorithm.finalizeCompute();

    /* Print the results */
    services::SharedPtr<pca::Result> result = algorithm.getResult();
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
