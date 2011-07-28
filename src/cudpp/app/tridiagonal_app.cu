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
/** @name Tridiagonal functions
 * @{
 */

#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cuda_util.h"

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "app/crpcr_app.cuh"

/**
 * @brief Dispatches the tridiagonal function based on the plan
 *
 * This is the dispatch call for the tridiagonal solver in either float 
 * or double datatype. 
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 * @param[in] plan pointer to CUDPPTridiagonalPlan
 * @returns CUDPPResult indicating success or error condition
 */
CUDPPResult cudppTridiagonalDispatch(void *d_a, 
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
        crpcr<float>((float *)d_a, 
                     (float *)d_b, 
                     (float *)d_c, 
                     (float *)d_d, 
                     (float *)d_x, 
                     systemSize, 
                     numSystems);
        return CUDPP_SUCCESS;
    }
    else if (plan->m_config.datatype == CUDPP_DOUBLE)
    {
        crpcr<double>((double *)d_a, 
                      (double *)d_b, 
                      (double *)d_c, 
                      (double *)d_d, 
                      (double *)d_x, 
                      systemSize, 
                      numSystems);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    
}

/** @} */ // end Tridiagonal functions
/** @} */ // end cudpp_app
