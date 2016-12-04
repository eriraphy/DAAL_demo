/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015-2016 Intel Corporation All Rights Reserved.
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
!    Intel(R) DAAL version information
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LIBRARY_VERSION_INFO"></a>
 * \example library_version_info.cpp
 */

#include "daal.h"
#include <iostream>

using namespace std;
using namespace daal::services;

int main(int argc, char *argv[])
{
    LibraryVersionInfo ver;

    std::cout << "Major version:          " << ver.majorVersion  << std::endl;
    std::cout << "Minor version:          " << ver.minorVersion  << std::endl;
    std::cout << "Update version:         " << ver.updateVersion << std::endl;
    std::cout << "Product status:         " << ver.productStatus << std::endl;
    std::cout << "Build:                  " << ver.build         << std::endl;
    std::cout << "Name:                   " << ver.name          << std::endl;
    std::cout << "Processor optimization: " << ver.processor     << std::endl;

    return 0;
}
