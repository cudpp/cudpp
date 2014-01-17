#include "skew.h"
#include <fstream>

using namespace std;
using namespace SA;
//using namespace mgpu;
typedef unsigned int uint;

int main(int argc, char** argv)
{
  //ContextPtr context = CreateCudaDevice(argc, argv, true);
GpuTimer Timer;
  if (argc!=2) cout << "Usage: ./exefile InputFile" << endl;
  else{
    vector<string> line_text(1000000); //this value is as large as 4294967295 to deal with 4GB data
    ifstream infile;
    infile.open(argv[1]);
    int idx = 0;
    while(!infile.eof())
    {
        getline(infile, line_text[idx++]);
    }
    idx--;
    int str_length = 0;
    for (int i = 0; i < idx; ++i)
    {
        str_length += line_text[i].length();
    }
    char* str = new char[str_length+4];
    unsigned int addr = 0;
    for (int i = 0; i < idx; ++i)
    {
        memcpy(str+addr, line_text[i].c_str(), sizeof(char)*line_text[i].length());
        addr+=line_text[i].length();
    }
    str[str_length] = '$';
    str[str_length+1] = '$';
    str[str_length+2] = '$';


    uint *keys_sa = new uint[str_length+1];
    uint *str_value= new uint [str_length+3];
    uint* d_str;
    uint* d_keys_sa;
/* size_t free_byte ;
 size_t total_byte ;
 cudaError_t  cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
 if ( cudaSuccess != cuda_status ){
   printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
   exit(EXIT_FAILURE);
 }
cout << "free mem=" << free_byte << endl;
cout << "total mem=" << total_byte <<endl; */ 
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_str, (str_length+3)*sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_sa, (str_length+1)*sizeof(uint)));
    //unsigned char *str_value= new unsigned char [str_length+3];
    for (int i=0;i<str_length;i++) str_value[i]=(uint) str[i];       
    for(int i=str_length;i<str_length+3;i++) str_value[i]=0;

    CUDA_SAFE_CALL(cudaMemcpy(d_str, str_value, (str_length+3)*sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(str_value, d_str, (str_length+3)*sizeof(uint), cudaMemcpyDeviceToHost));
/*ofstream myfile;
myfile.open("checkResult.txt");*/
  /* for (int i = 0; i < str_length+1; ++i)
    {
        cout << str[i];
    }
    cout << endl;
for (int i = 0; i < str_length+3; ++i)
    {
        cout << str_value[i] << " ";
    }
    cout << endl;*/
/*cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
 if ( cudaSuccess != cuda_status ){
   printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
   exit(EXIT_FAILURE);
 }
cout << "free mem=" << free_byte << endl;
cout << "total mem=" << total_byte <<endl; */
Timer.Start();
    runComputeSA(d_str, d_keys_sa, str_length);
Timer.Stop();

    CUDA_SAFE_CALL(cudaMemcpy(keys_sa, d_keys_sa, (str_length+1)*sizeof(uint), cudaMemcpyDeviceToHost));
cout << "Total time is " << Timer.ElapsedMillis() <<endl;   

/*    for (int i = 0; i < str_length+1; ++i)
    {
        for (int j = keys_sa[i]-1; j < str_length+1; ++j)
        {
            cout << str[j];
        }
        cout << keys_sa[i] <<endl;
    }
*/  
//myfile.close();
   printf("================ SA completed ====================\n");

    _SafeDeleteArray(keys_sa);
    _SafeDeleteArray(str);
    _SafeDeleteArray(str_value);
    CUDA_SAFE_CALL(cudaFree(d_keys_sa));
    CUDA_SAFE_CALL(cudaFree(d_str));
    return 0;
}
}
