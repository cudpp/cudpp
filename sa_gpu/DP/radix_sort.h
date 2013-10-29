#include "sa_util.h"

namespace SA
{

	class CStrRadixSortEngine
	{
	    public:
		CStrRadixSortEngine() {};
		~CStrRadixSortEngine() {};
		void KeysOnlySort(unsigned int numElem, unsigned int* h_keys);

        void KeyValueSort(unsigned int numElem, unsigned int* h_keys, unsigned int* h_values);
        //void KeyValueSort(unsigned int numElem, unsigned long long* h_keys, unsigned int* h_values);
	};

}
