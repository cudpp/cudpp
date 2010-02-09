// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * tridiagonal_gold.cpp
 *
 * @brief Host testrig routines for the tridiagonal solver
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include <iostream>
#include <fstream>
using namespace std;

template <class T>
void serial(T *a,T *b,T *c,T *d,T *x,int num_elements)
{
    c[num_elements-1]=0;
    c[0]=c[0]/b[0];
    d[0]=d[0]/b[0];

    for (int i = 1; i < num_elements; i++)
    {
      c[i]=c[i]/(b[i]-a[i]*c[i-1]);
      d[i]=(d[i]-d[i-1]*a[i])/(b[i]-a[i]*c[i-1]);  
    }

    x[num_elements-1]=d[num_elements-1];

    for (int i = num_elements-2; i >=0; i--)
    {
      x[i]=d[i]-c[i]*x[i+1];
    }    
}

template <class T>
void serial_small_systems(T *a, T *b, T *c, T *d, T *x, int system_size, int num_systems)
{
    for (int i = 0; i < num_systems; i++)
    {
        serial(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x[i*system_size],system_size);
    }
}

template <class T>
T rand01()
{
    return T(rand())/T(RAND_MAX);
}

template <class T>
void test_gen(T *a,T *b,T *c,T *d,T *x,int system_size)
{
    //generate a diagonally dominated matrix
    for (int j = 0; j < system_size; j++)
    {
        b[j] = 8 + rand01<T>();
        a[j] = 3 + rand01<T>();
        c[j] = 2 + rand01<T>();
        d[j] = 5 + rand01<T>();
        x[j] = 0;
    }      
    a[0] = 0;
    c[system_size-1] = 0;

}

template <class T>
void file_write_small_systems(T *x,int num_systems,int system_size, char *file_name)
{
    ofstream myfile;
    myfile.open (file_name);
    for(int i=0;i<num_systems*system_size;i++)
    {
        if (i%system_size==0)
            myfile << "***The following is the result of the equation set " << i/system_size << "\n";
        myfile << x[i] << "\n";
    }
    myfile.close();
}

template <class T>
T compare(T *x1, T *x2, int num_elements)
{
    T mean = 0;//mean error
    T root = 0;//root mean square error
    T max = 0; //max error

    for (int i = 0; i < num_elements; i++)
    {
        root += (x1[i] - x2[i]) * (x1[i] - x2[i]);
        mean += fabs(x1[i] - x2[i]);
        if(fabs(x1[i] - x2[i]) > max) max = fabs(x1[i] - x2[i]);
    }
    mean /= num_elements;
    root /= num_elements;
    root = sqrt(root); 

    return root;
}

template <class T>
int compare_small_systems(T *x1,T *x2, int system_size, int num_systems, const T epsilon)
{
    int retval = 0;

    for (int i = 0; i < num_systems; i++)
    {
        T diff = compare<T>(&x1[i*system_size],&x2[i*system_size],system_size);
        if(diff > epsilon || diff != diff) //if diff is QNAN/NAN, diff != diff will return true
        {
            cout<<"test failed, error is larger than " << epsilon << "\n";
            retval++;
        }
    }

    return retval;
}
