/* file: kmeans_dense_distr.cpp */
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
!    C++ example of dense K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_DENSE_DISTRIBUTED"></a>
 * \example kmeans_dense_distributed.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;
const size_t nBlocks     = 4;
const size_t nVectorsInBlock = 2500;

const string dataFileNames[] =
{
    "../data/distributed/kmeans_dense_1.csv", "../data/distributed/kmeans_dense_2.csv",
    "../data/distributed/kmeans_dense_3.csv", "../data/distributed/kmeans_dense_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &dataFileNames[0], &dataFileNames[1], &dataFileNames[2], &dataFileNames[3]);

    kmeans::Distributed<step2Master> masterAlgorithm(nClusters);

    NumericTablePtr centroids;
    NumericTablePtr assignments[nBlocks];
    NumericTablePtr goalFunction;

    kmeans::init::Distributed<step2Master,double,kmeans::init::randomDense> masterInit(nClusters);
    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i], DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        /* Create an algorithm object for the K-Means algorithm */
        kmeans::init::Distributed<step1Local,double,kmeans::init::randomDense> localInit(nClusters, nBlocks*nVectorsInBlock, i*nVectorsInBlock);

        localInit.input.set(kmeans::init::data, dataSource.getNumericTable());
        localInit.compute();

        masterInit.input.add(kmeans::init::partialResults, localInit.getPartialResult());
    }
    masterInit.compute();
    masterInit.finalizeCompute();
    centroids = masterInit.getResult()->get(kmeans::init::centroids);

    for(size_t it = 0; it < nIterations + 1; it++)
    {
        for (size_t i = 0; i < nBlocks; i++)
        {
            /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
            FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i], DataSource::doAllocateNumericTable,
                                                         DataSource::doDictionaryFromContext);

            /* Retrieve the data from the input file */
            dataSource.loadDataBlock();

            /* Create an algorithm object for the K-Means algorithm */
            kmeans::Distributed<step1Local> localAlgorithm(nClusters, it == nIterations);

            /* Set the input data to the algorithm */
            localAlgorithm.input.set(kmeans::data,           dataSource.getNumericTable());
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);

            localAlgorithm.compute();

            if( it == nIterations )
            {
                localAlgorithm.finalizeCompute();
                assignments[i] = localAlgorithm.getResult()->get(kmeans::assignments);
            }
            else
            {
                masterAlgorithm.input.add(kmeans::partialResults, localAlgorithm.getPartialResult());
            }
        }

        if( it == nIterations ) break;

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
        goalFunction = masterAlgorithm.getResult()->get(kmeans::goalFunction);
    }

    /* Print the clusterization results */
    printNumericTable(assignments[0], "First 10 cluster assignments from 1st node:", 10);
    printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(goalFunction,   "Goal function value:");

    return 0;
}
