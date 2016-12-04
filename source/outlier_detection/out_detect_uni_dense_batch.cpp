/* file: out_detect_uni_dense_batch.cpp */
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
!    C++ example of univariate outlier detection
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUTLIER_DETECTION_UNIVARIATE_BATCH"></a>
 * \example outlier_detection_univariate_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace algorithms;

typedef double inputDataFPType;     /* Input data floating-point type */

/* Input data set parameters */
string datasetFileName = "../data/batch/outlierdetection.csv";

struct UserInitialization : public univariate_outlier_detection::InitIface
{
    size_t nFeatures;

    explicit UserInitialization(size_t nFeatures) : nFeatures(nFeatures) {}

    virtual void operator()(NumericTable *data,
                            NumericTable *location,
                            NumericTable *scatter,
                            NumericTable *threshold)
    {

        BlockDescriptor<double> locationBlock;
        BlockDescriptor<double> scatterBlock;
        BlockDescriptor<double> thresholdBlock;

        location->getBlockOfRows(0, 1, writeOnly, locationBlock);
        scatter->getBlockOfRows(0, 1, writeOnly, scatterBlock);
        threshold->getBlockOfRows(0, 1, writeOnly, thresholdBlock);

        for(size_t i = 0; i < nFeatures; i++)
        {
            locationBlock.getBlockPtr()[i]  = 0.0;
            scatterBlock.getBlockPtr()[i]   = 1.0;
            thresholdBlock.getBlockPtr()[i] = 3.0;
        }

        location->releaseBlockOfRows(locationBlock);
        scatter->releaseBlockOfRows(scatterBlock);
        threshold->releaseBlockOfRows(thresholdBlock);
    }
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    size_t nFeatures = dataSource.getNumberOfColumns();

    univariate_outlier_detection::Batch<> algorithm;

    algorithm.input.set(univariate_outlier_detection::data, dataSource.getNumericTable());

    algorithm.parameter.initializationProcedure = services::SharedPtr<univariate_outlier_detection::InitIface>(new UserInitialization(
                                                                                                                   nFeatures));

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    services::SharedPtr<univariate_outlier_detection::Result> res = algorithm.getResult();

    printNumericTable(dataSource.getNumericTable(), "Input data");
    printNumericTable(res->get(univariate_outlier_detection::weights), "Outlier detection result (univariate)");

    return 0;
}
