/* file: error_handling_nothrow.cpp */
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
!    C++ example of error handling mechanism without throwing exceptions
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ERROR_HANDLING_NOTHROW"></a>
 * \example error_handling_nothrow.cpp
 */

#define DAAL_NOTHROW_EXCEPTIONS

#include "daal.h"
#include "service.h"

using namespace daal;

std::string wrongDatasetFileName = "../data/batch/wrong.csv";

int main(int argc, char *argv[])
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> wrongDataSource(wrongDatasetFileName);
    /* No exception was generated due to DAAL_NOTHROW_EXCEPTIONS define */

    /* Retrieve errors from FileDataSource<CSVFeatureManager> and their description. */
    std::cout << "FileDataSource error: " << wrongDataSource.getErrors()->getDescription() << std::endl;

    return 0;
}
