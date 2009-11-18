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
#include "cudpp_plan_manager.h"
#include "cudpp_maximal_launch.h"

typedef void* KernelPointer;

extern "C" size_t getNumCTAs(const CUDPPPlan* plan, KernelPointer kernel)
{
    return plan->m_planManager->numCTAs(kernel);    
}
extern "C" void compNumCTAs(const CUDPPPlan* plan, KernelPointer kernel, size_t bytesDynamicSharedMem, size_t threadsPerBlock)
{
    plan->m_planManager->computeNumCTAs(kernel, bytesDynamicSharedMem, threadsPerBlock);
}

/**
 * @brief Creates an instance of the CUDPP library, and returns a handle.
 */
CUDPP_DLL
CUDPPResult cudppCreate(CUDPPHandle* theCudpp)
{
    CUDPPPlanManager *mgr = new CUDPPPlanManager();
    *theCudpp = mgr->getHandle();
    return CUDPP_SUCCESS;
}

/**
 * @brief Destroys an instance of the CUDPP library given its handle.
 */
CUDPP_DLL
CUDPPResult cudppDestroy(CUDPPHandle theCudpp)
{
    CUDPPPlanManager *mgr = CUDPPPlanManager::getManagerFromHandle(theCudpp);
    delete mgr;
    mgr = 0;
    return CUDPP_SUCCESS;
}

//! @brief Plan Manager constructor
CUDPPPlanManager::CUDPPPlanManager()
{
}

/** @brief Plan Manager destructor 
* Destroys all plans as well as the plan manager.
*/
CUDPPPlanManager::~CUDPPPlanManager()
{
    numCTAsTable.clear();
}

/** @brief Retrieve the calculated maximal CTA launch size for the given kernel
  * This is used by CUDPP routines such as radix sort which perform a "maximal"
  * launch -- just enough CTAs to fill the device.
  */
size_t CUDPPPlanManager::numCTAs(KernelPointer kernel)
{
    return numCTAsTable[kernel];
}

/** @brief Calculate the maximal CTA launch size for the given kernel and resource usage.
  * This is used by CUDPP routines such as radix sort which perform a "maximal"
  * launch -- just enough CTAs to fill the device.
  */
void CUDPPPlanManager::computeNumCTAs(KernelPointer kernel, size_t bytesDynamicSharedMem, size_t threadsPerBlock)
{
    numCTAsTable[kernel] = maxBlocks(kernel, bytesDynamicSharedMem, threadsPerBlock);
}
