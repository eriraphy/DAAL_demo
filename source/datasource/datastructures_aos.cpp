/* file: datastructures_aos.cpp */
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
!    C++ example of using an array of structures (AOS)
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_AOS">
 * @example datastructures_aos.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

struct PointType
{
    float x;
    float y;
    int categ;
    double value;
};

int main()
{
    std::cout << "Array of structures (AOS) numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures = 4;
    PointType points[nObservations] =
    {
        {0.5f, -1.3f, 1, 100.1},
        {2.5f, -3.3f, 2, 200.2},
        {4.5f, -5.3f, 2, 350.3},
        {6.5f, -7.3f, 0, 470.4},
        {8.5f, -9.3f, 1, 270.5}
    };

    /* Create a new dictionary and fill it with the information about data */
    NumericTableDictionary newDict(nFeatures);

    /* Add a feature type to the dictionary */
    newDict[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict[2].featureType = data_feature_utils::DAAL_CATEGORICAL;
    newDict[3].featureType = data_feature_utils::DAAL_CONTINUOUS;

    /* Set the number of categories for a categorical feature */
    newDict[2].categoryNumber = 3;

    /* Construct AOS numericTable for a data array with nFeatures fields and nObservations elements*/
    AOSNumericTable dataTable(points, nFeatures, nObservations);

    /* Assign the new dictionary to an existing numeric table */
    dataTable.setDictionary(&newDict);

    /* Add data to the numeric table */
    dataTable.setFeature<float> (0, DAAL_STRUCT_MEMBER_OFFSET(PointType, x));
    dataTable.setFeature<float> (1, DAAL_STRUCT_MEMBER_OFFSET(PointType, y));
    dataTable.setFeature<int>   (2, DAAL_STRUCT_MEMBER_OFFSET(PointType, categ));
    dataTable.setFeature<double>(3, DAAL_STRUCT_MEMBER_OFFSET(PointType, value));

    /* Read a block of rows */
    const size_t firstReadRow = 0;

    BlockDescriptor<double> doubleBlock;
    dataTable.getBlockOfRows(firstReadRow, nObservations, readOnly, doubleBlock);
    printArray<double>(doubleBlock.getBlockPtr(), nFeatures, doubleBlock.getNumberOfRows(), "Print AOS data structures as double:");
    dataTable.releaseBlockOfRows(doubleBlock);

    /* Read a feature (column) */
    size_t readFeatureIdx = 2;

    BlockDescriptor<int> intBlock;
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, intBlock);
    printArray<int>(intBlock.getBlockPtr(), 1, intBlock.getNumberOfRows(), "Print the third feature of AOS:");
    dataTable.releaseBlockOfColumnValues(intBlock);

    return 0;
}
