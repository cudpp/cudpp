#include "sa_util.h"
#include "skew.cu"
#include <include/moderngpu.cuh>

using namespace std;
using namespace SA;
using namespace mgpu;

int main(int argc, char** argv)
{
  ContextPtr context = CreateCudaDevice(argc, argv, true);

  if (argc!=2) cout << "Usage: ./exefile InputFile" << endl;
  else{
    vector<string> line_text(1000000000); //this value is as large as 4294967295 to deal with 4GB data
    ifstream infile;
    infile.open(argv[1]);
    int idx = 0;
    while(!infile.eof())
    {
        getline(infile, line_text[idx++]);
    }
    idx--;
    unsigned int str_length = 0;
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
  /* for (int i = 0; i < str_length+1; ++i)
    {
        cout << str[i];
    }
    cout << endl;*/

    unsigned int *keys_sa = new unsigned int[str_length+1];
    unsigned int *str_value= new unsigned int [str_length+3];
    //unsigned char *str_value= new unsigned char [str_length+3];
    for (int i=0;i<str_length;i++) str_value[i]=(unsigned int) str[i];       
    for(int i=str_length;i<str_length+3;i++) str_value[i]=0;

    runComputeSA(str_value, keys_sa, str_length, *context);
   
   /* for (int i = 0; i < str_length+1; ++i)
    {
        for (int j = keys_sa[i]-1; j < str_length+1; ++j)
        {
            printf("%c", str[j]);
        }
        printf(" %d\n", keys_sa[i]);
    }*/
   printf("================ SA completed ====================\n");

    _SafeDeleteArray(keys_sa);
    _SafeDeleteArray(str);
    _SafeDeleteArray(str_value);
    return 0;
}
}
