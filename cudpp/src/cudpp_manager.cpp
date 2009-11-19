// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include "cudpp.h"
#include "cudpp_plan.h"
#include "cudpp_manager.h"
#include "cudpp_maximal_launch.h"

typedef void* KernelPointer;

/**
 * @brief Creates an instance of the CUDPP library, and returns a handle.
 */
CUDPP_DLL
CUDPPResult cudppCreate(CUDPPHandle* theCudpp)
{
    CUDPPManager *mgr = new CUDPPManager();
    *theCudpp = mgr->getHandle();
    return CUDPP_SUCCESS;
}

/**
 * @brief Destroys an instance of the CUDPP library given its handle.
 */
CUDPP_DLL
CUDPPResult cudppDestroy(CUDPPHandle theCudpp)
{
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(theCudpp);
    delete mgr;
    mgr = 0;
    return CUDPP_SUCCESS;
}

//! @brief CUDPP Manager constructor
CUDPPManager::CUDPPManager()
{
}

/** @brief CUDPP Manager destructor 
*/
CUDPPManager::~CUDPPManager()
{
}
