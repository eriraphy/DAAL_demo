/* file: assoc_rules_apriori_batch.cpp */
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
!    C++ example of association rules mining
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-APRIORI_BATCH"></a>
 * \example apriori_batch.cpp
 */

#include "daal.h"
#include "service.h"
using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "../data/batch/apriori.csv";

/* Apriori algorithm parameters */
const double minSupport     = 0.001;    /* Minimum support */
const double minConfidence  = 0.7;      /* Minimum confidence */

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to mine association rules using the Apriori method */
    association_rules::Batch<> algorithm;

    /* Set the input object for the algorithm */
    algorithm.input.set(association_rules::data, dataSource.getNumericTable());

    /* Set the Apriori algorithm parameters */
    algorithm.parameter.minSupport = minSupport;
    algorithm.parameter.minConfidence = minConfidence;

    /* Find large item sets and construct association rules */
    algorithm.compute();

    /* Get computed results of the Apriori algorithm */
    services::SharedPtr<association_rules::Result> res = algorithm.getResult();

    /* Print the large item sets */
    printAprioriItemsets(res->get(association_rules::largeItemsets),
                         res->get(association_rules::largeItemsetsSupport));

    /* Print the association rules */
    printAprioriRules(res->get(association_rules::antecedentItemsets),
                      res->get(association_rules::consequentItemsets),
                      res->get(association_rules::confidence));

    return 0;
}
