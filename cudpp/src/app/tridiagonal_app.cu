// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 4400 $
// $Date: 2008-08-04 10:58:14 -0700 (Mon, 04 Aug 2008) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * tridiagonal_app.cu
 *
 * @brief CUDPP application-level tridiagonal solver routine
 */

#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "kernel/pcr_kernel.cu"
#include "kernel/cr_kernel.cu"
#include "kernel/crpcr_kernel.cu"

#ifdef __cplusplus
extern "C"
{
#endif

/**@brief Dispatches the rand function based on the plan
 *
 * This is the dispatch call which looks at the algorithm specified in \a plan 
 * and calls the appropriate random number generation algorithm.  
 *
 * @param[out] d_out the array allocated on device memory where the random 
 * numbers will be stored
 * must be of type unsigned int
 * @param[in] numElements the number of elements in the array d_out
 * @param[in] plan pointer to CUDPPRandPlan which contains the algorithm to run
 */
void cudppTridiagonalDispatch(void *a, void *b, void *c, void *d, void *x, int system_size, int num_systems, const CUDPPTridiagonalPlan * plan)
{
    //switch to figure out which algorithm to run
    if (plan->m_config.datatype == CUDPP_FLOAT)
    {
        switch(plan->m_config.algorithm)
        {
            case CUDPP_TRIDIAGONAL_CR:
                cyclic_small_systems<float>((float *)a, (float *)b, (float *)c, (float *)d, (float *)x, system_size, num_systems);
                break;
            case CUDPP_TRIDIAGONAL_PCR:
                pcr_small_systems<float>((float *)a, (float *)b, (float *)c, (float *)d, (float *)x, system_size, num_systems);
                break;
            case CUDPP_TRIDIAGONAL_CRPCR:
                crpcr_small_systems<float>((float *)a, (float *)b, (float *)c, (float *)d, (float *)x, system_size, num_systems);
                break;
            default:
                break;
        }
    }
    else
    {
        switch(plan->m_config.algorithm)
        {
            case CUDPP_TRIDIAGONAL_CR:
                cyclic_small_systems<double>((double *)a, (double *)b, (double *)c, (double *)d, (double *)x, system_size, num_systems);
                break;
            case CUDPP_TRIDIAGONAL_PCR:
                pcr_small_systems<double>((double *)a, (double *)b, (double *)c, (double *)d, (double *)x, system_size, num_systems);
                break;
            case CUDPP_TRIDIAGONAL_CRPCR:
                crpcr_small_systems<double>((double *)a, (double *)b, (double *)c, (double *)d, (double *)x, system_size, num_systems);
                break;
            default:
                break;
        }
    }
}

#ifdef __cplusplus
}
#endif
/** @} */ // end rand_app
/** @} */ // end cudpp_app
