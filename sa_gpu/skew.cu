#include "sa_util.h"
#include "radix_sort.h"
#include "merge.cuh"
#include <include/moderngpu.cuh>
#include <fstream>

using namespace std;
using namespace SA;
using namespace mgpu;

bool UniqueRank(unsigned int*key, int size)
{
    for ( int i = 0; i < size-1; ++i )
    {
        if (key[i] == key[i+1])
            return false;
    }
    return true;
}


void ComputeSA(unsigned int* str, unsigned int* keys_sa, int str_length, CudaContext& context)
{
    int mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    int mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    int mod_3 = (str_length+1)/3;
//cout << "mod_1=" << mod_1 << ",mod_2=" << mod_2 << endl;
    unsigned int *keys_srt_12 = new unsigned int[mod_1+mod_2];
    unsigned int *keys_uint_12a = new unsigned int[mod_1+mod_2];
    unsigned int *keys_uint_12b = new unsigned int[mod_1+mod_2];
    unsigned int *keys_uint_12c = new unsigned int[mod_1+mod_2];
    unsigned int *keys_sa_12 = new unsigned int[str_length+2];
    unsigned int *rank =new unsigned int[mod_1+mod_2];
    float timer=0;
  printf("================in SA===================\n");  

    //initialize substring tuple
    for ( int i = 0; i < mod_1; ++i )
    {
        keys_srt_12[i] = i*3+1;
//cout << (unsigned int)str[i*3] << ',' <<  (unsigned int)str[i*3+1] << ',' <<  (unsigned int)str[3*i+2] << ',' <<  keys_srt_12[i] <<endl;
        keys_uint_12c[i] = str[i*3+2];

    }
    for ( int i = mod_1; i < mod_1 + mod_2; ++i )
    {
        keys_srt_12[i] = (i-mod_1)*3+2;
//cout << (unsigned int)str[(i-mod_1)*3+1] << ',' << (unsigned int)str[(i-mod_1)*3+2] << ',' << (unsigned int)str[(i-mod_1)*3+3] << ',' << keys_srt_12[i] << endl;
         
        keys_uint_12c[i] = str[(i-mod_1)*3+3];

    }
GpuTimer Timer1;
//radix sort using constructed key
    CStrRadixSortEngine sorter;
Timer1.Start();
    sorter.KeyValueSort(mod_1 + mod_2, keys_uint_12c, keys_srt_12);
Timer1.Stop();
timer += Timer1.ElapsedMillis();
for (int i = 0; i < mod_1+mod_2; ++i)
        {
            //cout << (unsigned int)str[keys_srt_12[i]-1] << ',' <<  keys_uint_12c[i] << ' ' << keys_srt_12[i] << '|';
            keys_uint_12b[i] = str[keys_srt_12[i]];

         }
    // printf("\n");
_SafeDeleteArray(keys_uint_12c);
GpuTimer Timer2;
Timer2.Start();
    sorter.KeyValueSort(mod_1 + mod_2, keys_uint_12b, keys_srt_12);
Timer2.Stop();
timer += Timer2.ElapsedMillis();
for (int i = 0; i < mod_1+mod_2; ++i)
        {
           // cout << (unsigned int)str[keys_srt_12[i]-1] << ',' <<  keys_uint_12b[i] << ' ' << keys_srt_12[i] << '|';
            keys_uint_12a[i] = str[keys_srt_12[i]-1];

         }
_SafeDeleteArray(keys_uint_12b);
    // printf("\n");
GpuTimer Timer3;
Timer3.Start();
    sorter.KeyValueSort(mod_1 + mod_2, keys_uint_12a, keys_srt_12);
Timer3.Stop();
timer += Timer3.ElapsedMillis();
_SafeDeleteArray(keys_uint_12a);
/*for (int i = 0; i < mod_1+mod_2; ++i)
        {
            cout << (unsigned int)str[keys_srt_12[i]-1] << ',' <<  keys_uint_12a[i] << ' ' << keys_srt_12[i] << '|';

         }
     printf("\n");*/
cout << "--------------sorter12 completed-------------------------" <<endl;
   

   /* ofstream myfile1;
    myfile1.open("checkSA12.txt");
 
    for (int i = 0; i < mod_1+mod_2; ++i)
        {
            myfile1 << (unsigned int)str[keys_srt_12[i]-1] << ',' <<  keys_uint_12[i] << ' ' << keys_srt_12[i] << '|';

         }
    myfile1.close();*/
     //printf("\n");
   
    int j;
    for(int i=0;i<mod_1+mod_2;i++)
       {
        
            if (i==0) {j=1; rank[i]=j; /*cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/}
            else if( (str[keys_srt_12[i]-1]==str[keys_srt_12[i-1]-1]) && (str[keys_srt_12[i]]==str[keys_srt_12[i-1]]) && (str[keys_srt_12[i]+1]==str[keys_srt_12[i-1]+1]) ) { rank[i]=j;   /* cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/ }
            else {j++;  rank[i]=j;   /* cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/}
   //keys_srt_12[i]: sorted index       
       }
    
//for(int i=0;i <mod_1+mod_2; ++i) cout << rank[i] << "|" ; cout << endl;

   if (!UniqueRank(rank,mod_1+mod_2))
{  
     
  //unsigned char *keys_rank_12=new unsigned char[mod_1+mod_2+3];
  unsigned int *keys_rank_12=new unsigned int[mod_1+mod_2+3];
  
    int j;
    for(int i=0;i<mod_1+mod_2;i++)
       {
        
            if (i==0) {j=1; keys_sa_12[keys_srt_12[i]-1]=j; /*cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/}
            else if(  (str[keys_srt_12[i]-1]==str[keys_srt_12[i-1]-1]) && (str[keys_srt_12[i]]==str[keys_srt_12[i-1]]) && (str[keys_srt_12[i]+1]==str[keys_srt_12[i-1]+1]) ) {keys_sa_12[keys_srt_12[i]-1]=j;    /* cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/}
            else {j++; keys_sa_12[keys_srt_12[i]-1]=j;   /* cout << keys_srt_12[i] << "'s rank is " << j <<endl;*/}
   //keys_srt_12[i]: sorted index       
       }

    for(int i=0;i<mod_1;i++) keys_rank_12[i]=keys_sa_12[i*3];
    for(int i=mod_1;i<mod_1+mod_2;i++) keys_rank_12[i]=keys_sa_12[(i-mod_1)*3+1];
    keys_rank_12[mod_1+mod_2]=0;
    keys_rank_12[mod_1+mod_2+1]=0;
    ComputeSA(keys_rank_12,keys_srt_12,mod_1+mod_2-1,context);

   for (int i=0;i<mod_1+mod_2;i++)
   {
     if(keys_srt_12[i]>mod_1) keys_srt_12[i]=3*(keys_srt_12[i]-mod_1-1)+2;
     else keys_srt_12[i]=3*(keys_srt_12[i]-1)+1;
   }
 
        _SafeDeleteArray(keys_rank_12);
       
}

    unsigned int *keys_uint_3a = new unsigned int[mod_3];
    unsigned int *keys_uint_3b = new unsigned int[mod_3];
    unsigned int *keys_srt_3 = new unsigned int[mod_3];

    for ( int i = 0; i < mod_1+mod_2; ++i )
    {
        keys_sa_12[keys_srt_12[i]] = i;
      
    }
    // cout <<endl;
   //  keys_sa_12: sorted mod12 ranks
   
    for (int i = 0; i < mod_3; ++i)
    {

        keys_srt_3[i] = i*3+3; 
  // cout << (unsigned int)str[i*3+2] << ',' <<  (unsigned int)str[i*3+3] << ',' <<  (unsigned int)str[3*i+4] << ',' <<  keys_srt_3[i] <<endl;
      
        keys_uint_3b[i] = (mod_1+mod_2+mod_3-(i*3+3)>0) ? keys_sa_12[i*3+4] : 0;

    }
    //radix sort mod3=0
GpuTimer Timer4;
Timer4.Start();
    sorter.KeyValueSort(mod_3, keys_uint_3b, keys_srt_3);
Timer4.Stop();
timer += Timer4.ElapsedMillis();
for (int i = 0; i < mod_3; ++i)
{
//cout << (unsigned int)str[keys_srt_3[i]-1] << ',' << keys_uint_3b[i] << ' ' << keys_srt_3[i] << '|';
    keys_uint_3a[i] = str[keys_srt_3[i]-1];
}
//printf("\n");
GpuTimer Timer5;
Timer5.Start();
    sorter.KeyValueSort(mod_3, keys_uint_3a, keys_srt_3);
Timer5.Stop();
timer += Timer5.ElapsedMillis();
/*for (int i = 0; i < mod_3; ++i)
{
cout << (unsigned int)str[keys_srt_3[i]-1] << ',' << keys_uint_3a[i] << ' ' << keys_srt_3[i] << '|';
}
printf("\n");*/

  cout << "---------------sorter3 completed------------------------" <<endl;
       
    
//cout << "mod_3=" << mod_3 << endl;
  /*  ofstream myfile2;
    myfile2.open("checkSA3.txt");
for (int i = 0; i < mod_3; ++i)
{
myfile2 << (unsigned int)str[keys_srt_3[i]-1] << ',' << keys_uint_3[i] << ' ' << keys_srt_3[i] << '|';
}
   myfile2.close();*/
//printf("\n");

//////////////////////////// merge sort//////////////////////////////////
    Vector* aKeysHost = new Vector[mod_1+mod_2];
    Vector* bKeysHost = new Vector[mod_3];
    int* aValsHost=new int[mod_1+mod_2];
    int* bValsHost=new int[mod_3];
    Vector* cKeysHost=new Vector[mod_1+mod_2+mod_3];
    int* cValsHost=new int[mod_1+mod_2+mod_3];
    int aCount=mod_1+mod_2;
    int bCount=mod_3;
    int bound = aCount+bCount;

    for(int i=0; i< mod_1+mod_2; ++i)
    {
       
       if(keys_srt_12[i]%3==1)  
            {
              aKeysHost[i].a = str[keys_srt_12[i]-1];
              aKeysHost[i].b = (bound-keys_srt_12[i]>0) ? keys_sa_12[keys_srt_12[i]+1] : 0;
              aKeysHost[i].c = 0;
              aKeysHost[i].d = 1;
            }
       else  
        
            {
              aKeysHost[i].a = str[keys_srt_12[i]-1];
              aKeysHost[i].b = (bound-keys_srt_12[i]>0) ? str[keys_srt_12[i]] : 0;
              aKeysHost[i].c = (bound-keys_srt_12[i]>1) ? keys_sa_12[keys_srt_12[i]+2] : 0;
              aKeysHost[i].d = 0;
            }

     }


    for(int j=0; j< mod_3; ++j)
    {
     
        bKeysHost[j].a = str[keys_srt_3[j]-1];
        bKeysHost[j].b = (bound-keys_srt_3[j]>0) ? str[keys_srt_3[j]] : 0;
        bKeysHost[j].c = (bound-keys_srt_3[j]>0) ? keys_sa_12[keys_srt_3[j]+1] : 0;
        bKeysHost[j].d = (bound-keys_srt_3[j]>1) ? keys_sa_12[keys_srt_3[j]+2] : 0;

    }
    // aVals=keys_srt_12; bVals=keys_srt_3;
    
     for(int i=0;i<mod_1+mod_2;i++) aValsHost[i]=keys_srt_12[i];
     for(int j=0;j<mod_3;j++) bValsHost[j]=keys_srt_3[j];//bValsHost[j+mod_3]=keys_srt_3[j];}

     MGPU_MEM(Vector) aKeys=context.Malloc((const Vector*) aKeysHost, (size_t) aCount);
     MGPU_MEM(int) aVals=context.Malloc((const int*) aValsHost, (size_t) aCount);
    // cout << "-------------a Keys and Vals mallocated -------------------------" <<endl;
     MGPU_MEM(Vector) bKeys=context.Malloc((const Vector*) bKeysHost, (size_t) bCount);
    // cout << "---------------bKeys mallocated -------------------------" <<endl;
     MGPU_MEM(int) bVals=context.Malloc((const int*) bValsHost, (size_t) bCount);
     //cout << "---------------bVals mallocated -------------------------" <<endl;    
     MGPU_MEM(Vector) cKeys=context.Malloc<Vector>(mod_1+mod_2+mod_3);   
     MGPU_MEM(int) cVals=context.Malloc<int>(mod_1+mod_2+mod_3);
     
     //cout << "------------------------merge-------------------------" <<endl;
    // cout << "aCount=" << aCount << "," << "bCount=" << bCount <<endl;
    // cout << "SA12 sorted positions" <<endl;
    // for(int i=0;i<aCount;i++) cout << keys_srt_12[i] << " "; cout <<endl;
    /* cout << "aKeys:" <<endl;
     for(int i=0;i<aCount;i++) cout << aKeysHost[i].a << "," << aKeysHost[i].b << "," << aKeysHost[i].c << "," << aKeysHost[i].d << "|"; cout <<endl;
     cout << "aVals:" <<endl;
     for(int i=0;i<aCount;i++) cout << aValsHost[i] << " "; cout <<endl;
     cout << "bKeys:" <<endl;
     for(int j=0;j<bCount;j++) cout << bKeysHost[j].a << "," << bKeysHost[j].b << "," << bKeysHost[j].c << "," << bKeysHost[j].d << "|"; cout <<endl;
     cout << "bVals:" <<endl;
     for(int j=0;j<bCount;j++) cout << bValsHost[j] << " "; cout <<endl;*/
     
     
     printf("aCount %d bCount %d cCount %d\n", aCount, bCount, mod_1+mod_2+mod_3);
GpuTimer Timer6;
Timer6.Start();
     MergePairs(aKeys->get(), aVals->get(), aCount, bKeys->get(), bVals->get(), bCount, cKeys->get(), cVals->get(), context);
Timer6.Stop();
     timer += Timer6.ElapsedMillis();
     cout << "Total time is " << timer <<endl;
   // cout << "------merged----------" <<endl;
      cKeys->ToHost(cKeysHost,(size_t)(mod_1+mod_2+mod_3));
    // cout << "------------------cKeys to Host successfully ------------------" <<endl;
      cVals->ToHost(cValsHost,(size_t)(mod_1+mod_2+mod_3));
   // ofstream myfile;
   // myfile.open("check.txt");
   
     //for(int i=0;i<mod_1+mod_2+mod_3;i++) {keys_sa[i] = cValsHost[i]; cout << "cVals=" << cValsHost[i] << ", " << "cKeys.x=" << cKeysHost[i].x << "  cKeys.y=" << cKeysHost[i].y <<endl; } cout << endl;
     for(int i=0;i<mod_1+mod_2+mod_3;i++) keys_sa[i] = cValsHost[i]; //myfile << "cVals=" << cValsHost[i] << ", " << "cKeys.a=" << cKeysHost[i].a << "  cKeys.b=" << cKeysHost[i].b << " cKeys.c=" << cKeysHost[i].c << " cKeys.d=" << cKeysHost[i].d <<endl; }// cout << endl;
//    myfile.close();



    _SafeDeleteArray(rank);
    _SafeDeleteArray(aKeysHost);
    _SafeDeleteArray(aValsHost);
    _SafeDeleteArray(bKeysHost);
    _SafeDeleteArray(bValsHost);
    _SafeDeleteArray(cKeysHost);
    _SafeDeleteArray(cValsHost);
    //_SafeDeleteArray(keys_uint_12a);
    //_SafeDeleteArray(keys_uint_12b);
    //_SafeDeleteArray(keys_uint_12c);
    _SafeDeleteArray(keys_srt_12);
    _SafeDeleteArray(keys_uint_3a);
    _SafeDeleteArray(keys_uint_3b);    
    _SafeDeleteArray(keys_srt_3);
    _SafeDeleteArray(keys_sa_12);

}
