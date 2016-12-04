/* file: pca_svd_dense_distr.cpp */
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
!    C++ example of principal component analysis (PCA) using the singular value
!    decomposition (SVD) method in the distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_SVD_DENSE_DISTRIBUTED"></a>
 * \example pca_svd_dense_distributed.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks         = 4;
const size_t nVectorsInBlock = 250;
size_t nFeatures;

const string dataFileNames[] =
{
    "../data/distributed/pca_normalized_1.csv", "../data/distributed/pca_normalized_2.csv",
    "../data/distributed/pca_normalized_3.csv", "../data/distributed/pca_normalized_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &dataFileNames[0], &dataFileNames[1], &dataFileNames[2], &dataFileNames[3]);

    /* Create an algorithm for principal component analysis using the SVD method on the master node */
    pca::Distributed<step2Master, double, pca::svdDense> masterAlgorithm;

    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i], DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

        /* Retrieve the input data */
        dataSource.loadDataBlock(nVectorsInBlock);

        /* Create an algorithm for principal component analysis using the SVD method on the local node */
        pca::Distributed<step1Local, double, pca::svdDense> localAlgorithm;

        /* Set the input data to the algorithm */
        localAlgorithm.input.set(pca::data, dataSource.getNumericTable());

        /* Compute PCA decomposition */
        localAlgorithm.compute();

        /* Set local partial results as input for the master-node algorithm */
        masterAlgorithm.input.add(pca::partialResults, localAlgorithm.getPartialResult());
    }

    /* Merge and finalize PCA decomposition on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    services::SharedPtr<pca::Result> result = masterAlgorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
