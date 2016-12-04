/* file: impl_als_csr_batch.cpp */
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
!    C++ example of the implicit alternating least squares (ALS) algorithm in
!    the batch processing mode.
!
!    The program trains the implicit ALS model on a training data set.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-IMPLICIT_ALS_CSR_BATCH"></a>
 * \example implicit_als_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters */
string trainDatasetFileName            = "../data/batch/implicit_als_csr.csv";

typedef double  algorithmFPType;        /* Algorithm floating-point type */
typedef double  dataFPType;             /* Input data floating-point type */

/* Algorithm parameters */
const size_t nFactors = 2;

NumericTablePtr dataTable;
services::SharedPtr<Model> initialModel;
services::SharedPtr<training::Result> trainingResult;

void initializeModel();
void trainModel();
void testModel();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    initializeModel();

    trainModel();

    testModel();

    return 0;
}

void initializeModel()
{
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    dataTable = NumericTablePtr(createSparseTable<dataFPType>(trainDatasetFileName));

    /* Create an algorithm object to initialize the implicit ALS model with the default method */
    training::init::Batch<algorithmFPType, training::init::fastCSR> initAlgorithm;
    initAlgorithm.parameter.nFactors = nFactors;

    /* Pass a training data set and dependent values to the algorithm */
    initAlgorithm.input.set(training::init::data, dataTable);

    /* Initialize the implicit ALS model */
    initAlgorithm.compute();

    initialModel = initAlgorithm.getResult()->get(training::init::model);
}

void trainModel()
{
    /* Create an algorithm object to train the implicit ALS model with the default method */
    training::Batch<algorithmFPType, training::fastCSR> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, dataTable);
    algorithm.input.set(training::inputModel, initialModel);

    algorithm.parameter.nFactors = nFactors;

    /* Build the implicit ALS model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Create an algorithm object to predict recommendations of the implicit ALS model */
    prediction::ratings::Batch<> algorithm;
    algorithm.parameter.nFactors = nFactors;

    algorithm.input.set(prediction::ratings::model, trainingResult->get(training::model));

    algorithm.compute();

    NumericTablePtr predictedRatings = algorithm.getResult()->get(prediction::ratings::prediction);

    printNumericTable(predictedRatings, "Predicted ratings:");
}
