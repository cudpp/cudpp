#include "sa_util.h"

namespace SA
{

	class CStrRadixSortEngine
	{
	    public:
		CStrRadixSortEngine() {};
		~CStrRadixSortEngine() {};
                void KeyValueSort(unsigned int numElem, unsigned int* d_keys, unsigned int* d_values);
	};

}
