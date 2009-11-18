// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_PLAN_MANAGER_H__
#define __CUDPP_PLAN_MANAGER_H__

#include <map>

class CUDPPPlan;
typedef void* KernelPointer;

/** @brief Singleton manager class for CUDPPPlan objects
  * 
  * This class manages resources used by CUDPP plans.  
  *
  * Currently it only manages a table containing maximal CTA
  * counts for certain kernels that perform maximal launches.
  */
class CUDPPPlanManager
{
public:

    CUDPPPlanManager();
    ~CUDPPPlanManager();
   
    size_t      numCTAs(KernelPointer kernel);
    void        computeNumCTAs(KernelPointer kernel, 
                               size_t bytesDynamicSharedMem, 
                               size_t threadsPerBlock);

    //! @internal Convert an opaque handle to a pointer to a plan manager
    //! @param [in] cudppHandle Handle to the Plan Manager object
    static CUDPPPlanManager* getManagerFromHandle(CUDPPHandle cudppHandle)
    {
        return reinterpret_cast<CUDPPPlanManager*>(cudppHandle);
    }

    //! @internal Convert an opaque handle to a pointer to a plan manager
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
    
protected:
    std::map<void*, size_t> numCTAsTable;
};

#endif // __CUDPP_PLAN_MANAGER_H__
