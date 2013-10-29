#include "sa_util.h"

#if defined(__CUDA__ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void) (f, __VA_ARGS__), 0)
#endif

using namespace SA;
//Really naive mergesort only for the course project

__global__ void SimpleMerge(int* str, unsigned int* ref_sa, unsigned int* value, int leftSize, int rightSize, unsigned int* res_val)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < leftSize+rightSize)
    {
        int rightPair = (index < leftSize) ? 0 : 1;
        int startPos = 0 + (1-rightPair)*leftSize;
        int endPos = leftSize - 1 + (1-rightPair)*rightSize;
        int idx = (int)value[index];
        int curPos = startPos;
        if (rightPair)
        {
            int rank_1 = idx+1;
            int rank_2 = idx+2;
            int pos_1 = (rank_1 > leftSize+rightSize) ? -1 : (int)ref_sa[rank_1];
            int pos_2 = (rank_2 > leftSize+rightSize) ? -1 : (int)ref_sa[rank_2];
            // Binary search to find rank
            while (curPos <= endPos)
            {
                bool flag = true;
                if (str[value[curPos]-1] > str[idx-1])
                    break;
                else if (str[value[curPos]-1] == str[idx-1])
                {
                    if (value[curPos]%3 == 1)
                    {
                        int rank_cur = value[curPos]+1;
                        int pos_cur = (rank_cur > leftSize+rightSize) ? -1 : (int)ref_sa[rank_cur];
                        if (pos_cur >= pos_1)
                        {
                            flag = false;
                            break;
                        }
                    }
                    else
                    {
                        if (str[value[curPos]] > str[idx])
                            break;
                        else if (str[value[curPos]] == str[idx])
                        {
                            int rank_cur = value[curPos]+2;
                            int pos_cur = (rank_cur > leftSize+rightSize) ? -1 : (int)ref_sa[rank_cur];
                            if (pos_cur > pos_2)
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                }
                if (!flag)
                    break;
                else
                {
                    curPos++;
                }
            }
        }
        else
        {
            if (idx%3 == 1)
            {
                int rank_1 = idx+1;
                int pos_1 = (rank_1 > leftSize+rightSize) ? -1 : (int)ref_sa[rank_1];
                // Binary search to find rank
                while (curPos <= endPos)
                {
                    bool flag = true;
                    if (str[value[curPos]-1] > str[idx-1])
                        break;
                    else if (str[value[curPos]-1] == str[idx-1])
                    {
                        int rank_cur = value[curPos]+1;
                        int pos_cur = (rank_cur > leftSize+rightSize) ? -1 : (int)ref_sa[rank_cur];
                        if (pos_cur >= pos_1)
                        {
                            flag = false;
                            break;
                        }
                    }
                    if (!flag)
                        break;
                    else
                    curPos++;
                }
            }
            if (idx%3 == 2)
            {
                int rank_2 = idx+2;
                int pos_2 = (rank_2 > leftSize+rightSize) ? -1 : (int)ref_sa[rank_2];

                while (curPos <= endPos)
                {
                    bool flag = true;
                    if (str[value[curPos]-1] > str[idx-1])
                        break;
                    else if (str[value[curPos]-1] == str[idx-1])
                    {
                        if (str[value[curPos]] > str[idx])
                            break;
                        else if (str[value[curPos]] == str[idx])
                        {
                            int rank_cur = value[curPos]+2;
                            int pos_cur = (rank_cur > leftSize+rightSize) ? -1 : (int)ref_sa[rank_cur];
                            if (pos_cur > pos_2)
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                    if (!flag)
                        break;
                    else
                    curPos++;
                }
            }
        }
        
        unsigned int rank = (curPos - leftSize*(1-rightPair)) + (index - leftSize*rightPair);
        res_val[rank] = value[index];
    }
}



















