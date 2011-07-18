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
 * @brief CUDPP application-level tridiagonal solver routines
 */

/** \addtogroup cudpp_app
  * @{
  */
/** @name Tridiagonal dispatch function
 * @{
 */

#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cuda_util.h"

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "app/cr_app.cu"
#include "app/pcr_app.cu"
#include "app/crpcr_app.cu"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Dispatches the tridiagonal function based on the plan
 *
 * This is the dispatch call which looks at the algorithm and datatype 
 * specified in \a plan, and calls the appropriate tridiagonal system 
 * solver. There are three algorithms available to choose from, which are 
 * cyclic reduction (CR), parallel cyclic reduction (PCR), and the hybrid 
 * CR-PCR  algorithm. Both float and double are supported datatypes.

 * @param[out] x Solution vector
 * @param[in] a Lower diagonal
 * @param[in] b Main diagonal
 * @param[in] c Upper diagonal
 * @param[in] d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 * @param[in] plan pointer to CUDPPTridiagonalPlan
 */
void cudppTridiagonalDispatch(void *d_a, 
                              void *d_b, 
                              void *d_c, 
                              void *d_d, 
                              void *d_x, 
                              int systemSize, 
                              int numSystems, 
                              const CUDPPTridiagonalPlan * plan)
{
  
    //figure out which algorithm to run
    if (plan->m_config.datatype == CUDPP_FLOAT)
    {
        switch(plan->m_config.options)
        {
            case CUDPP_OPTION_TRIDIAGONAL_CR:
                cr<float>((float *)d_a, 
                          (float *)d_b, 
                          (float *)d_c, 
                          (float *)d_d, 
                          (float *)d_x, 
                          systemSize, 
                          numSystems);
                break;
            case CUDPP_OPTION_TRIDIAGONAL_PCR:
                pcr<float>((float *)d_a, 
                           (float *)d_b, 
                           (float *)d_c, 
                           (float *)d_d, 
                           (float *)d_x, 
                           systemSize, 
                           numSystems);
                break;
            case CUDPP_OPTION_TRIDIAGONAL_CRPCR:
                crpcr<float>((float *)d_a, 
                             (float *)d_b, 
                             (float *)d_c, 
                             (float *)d_d, 
                             (float *)d_x, 
                             systemSize, 
                             numSystems);
                break;
            default:
                break;
        }
    }
    else if (plan->m_config.datatype == CUDPP_DOUBLE)
    {
        switch(plan->m_config.options)
        {
            case CUDPP_OPTION_TRIDIAGONAL_CR:
                cr<double>((double *)d_a, 
                           (double *)d_b, 
                           (double *)d_c, 
                           (double *)d_d, 
                           (double *)d_x, 
                           systemSize, 
                           numSystems);
                break;
            case CUDPP_OPTION_TRIDIAGONAL_PCR:
                pcr<double>((double *)d_a, 
                            (double *)d_b, 
                            (double *)d_c, 
                            (double *)d_d, 
                            (double *)d_x, 
                            systemSize, 
                            numSystems);
                break;
            case CUDPP_OPTION_TRIDIAGONAL_CRPCR:
                crpcr<double>((double *)d_a, 
                              (double *)d_b, 
                              (double *)d_c, 
                              (double *)d_d, 
                              (double *)d_x, 
                              systemSize, 
                              numSystems);
                break;
            default:
                break;
        }
    }
    else
        printf("datatype not specified\n");
    
}

#ifdef __cplusplus
}
#endif
/** @} */ // end Tridiagonal dispatch function
/** @} */ // end cudpp_app
