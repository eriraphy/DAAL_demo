/* file: svd_dense_distr.cpp */
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
!    C++ example of singular value decomposition (SVD) in the distributed
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVD_DISTRIBUTED"></a>
 * \example svd_distributed.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks      = 4;

const string datasetFileNames[] =
{
    "../data/distributed/svd_1.csv",
    "../data/distributed/svd_2.csv",
    "../data/distributed/svd_3.csv",
    "../data/distributed/svd_4.csv"
};

void computestep1Local(size_t block);
void computeOnMasterNode();
void finalizeComputestep1Local(size_t block);

DataCollectionPtr dataFromStep1ForStep2[nBlocks];
DataCollectionPtr dataFromStep1ForStep3[nBlocks];
DataCollectionPtr dataFromStep2ForStep3[nBlocks];
NumericTablePtr Sigma;
NumericTablePtr V    ;
NumericTablePtr Ui[nBlocks];

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++)
    {
        computestep1Local(i);
    }

    computeOnMasterNode();

    for (size_t i = 0; i < nBlocks; i++)
    {
        finalizeComputestep1Local(i);
    }

    /* Print the results */
    printNumericTable(Sigma, "Singular values:");
    printNumericTable(V,     "Right orthogonal matrix V:");
    printNumericTable(Ui[0], "Part of left orthogonal matrix U from 1st node:", 10);

    return 0;
}

void computestep1Local(size_t block)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[block], DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the input data */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute SVD on the local node */
    svd::Distributed<step1Local> algorithm;

    algorithm.input.set( svd::data, dataSource.getNumericTable() );

    /* Compute SVD */
    algorithm.compute();

    dataFromStep1ForStep2[block] = algorithm.getPartialResult()->get( svd::outputOfStep1ForStep2 );
    dataFromStep1ForStep3[block] = algorithm.getPartialResult()->get( svd::outputOfStep1ForStep3 );
}

void computeOnMasterNode()
{
    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step2Master> algorithm;

    for (size_t i = 0; i < nBlocks; i++)
    {
        algorithm.input.add( svd::inputOfStep2FromStep1, i, dataFromStep1ForStep2[i] );
    }

    /* Compute SVD */
    algorithm.compute();

    services::SharedPtr<svd::DistributedPartialResult> pres = algorithm.getPartialResult();
    KeyValueDataCollectionPtr inputForStep3FromStep2 = pres->get( svd::outputOfStep2ForStep3 );

    for (size_t i = 0; i < nBlocks; i++)
    {
        dataFromStep2ForStep3[i] = services::staticPointerCast<DataCollection, SerializationIface>((*inputForStep3FromStep2)[i]);
    }

    services::SharedPtr<svd::Result> res = algorithm.getResult();

    Sigma = res->get(svd::singularValues     );
    V     = res->get(svd::rightSingularMatrix);
}

void finalizeComputestep1Local(size_t block)
{
    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step3Local> algorithm;

    algorithm.input.set( svd::inputOfStep3FromStep1, dataFromStep1ForStep3[block] );
    algorithm.input.set( svd::inputOfStep3FromStep2, dataFromStep2ForStep3[block] );

    /* Compute SVD */
    algorithm.compute();

    algorithm.finalizeCompute();

    services::SharedPtr<svd::Result> res = algorithm.getResult();

    Ui[block] = res->get(svd::leftSingularMatrix);
}
