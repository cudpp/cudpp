////////////////////////////////////////////////////
// Include files and defines for b40c radix sort
////////////////////////////////////////////////////
#include <b40c/util/error_utils.cuh>
#include <b40c/util/ping_pong_storage.cuh>
#include <b40c/radix_sort/enactor.cuh>
#include "radix_sort.h"

using namespace b40c;

namespace SA
{
    template<
        radix_sort::ProbSizeGenre GENRE,
        typename PingPongStorage,
        typename SizeT>
            void KeySort(
                    PingPongStorage		&device_storage,
                    SizeT			num_elements,
                    typename PingPongStorage::KeyType *h_keys
                    )
            {
                typename PingPongStorage::KeyType K;

                // Create sorting enactor
                radix_sort::Enactor sorting_enactor;

                // Move a fresh copy of the problem into device storage
                cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice);

                sorting_enactor.template Sort<GENRE>(device_storage, num_elements, 0);

                // Copy out data
                cudaMemcpy(h_keys, device_storage.d_keys[device_storage.selector], sizeof(K) * num_elements, cudaMemcpyDeviceToHost);

            }

    template<
        radix_sort::ProbSizeGenre GENRE,
        typename PingPongStorage,
        typename SizeT>
            void KeysValueSort(
                    PingPongStorage     &device_storage,
                    SizeT               num_elements,
                    typename PingPongStorage::KeyType *h_keys,
                    typename PingPongStorage::ValueType *h_values
                    )
            {
                typename PingPongStorage::KeyType K;
                typename PingPongStorage::ValueType V;

                // Create sorting enactor
                radix_sort::Enactor sorting_enactor;

                // Move a fresh copy of the problem into device storage
                cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice);
                cudaMemcpy(device_storage.d_values[0], h_values, sizeof(V) * num_elements, cudaMemcpyHostToDevice);

                sorting_enactor.template Sort<GENRE>(device_storage, num_elements, 0);

                // Copy out data
                cudaMemcpy(h_keys, device_storage.d_keys[device_storage.selector], sizeof(K) * num_elements, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_values, device_storage.d_values[device_storage.selector], sizeof(V) * num_elements, cudaMemcpyDeviceToHost);

            }
    void CStrRadixSortEngine::KeysOnlySort(unsigned int numElem, unsigned int* h_keys)
    {
        util::PingPongStorage<unsigned int> device_storage;
        cudaMalloc((void**) &device_storage.d_keys[0], sizeof(unsigned int) * numElem);

        KeySort<radix_sort::LARGE_SIZE>(device_storage, numElem, h_keys);

        if ( device_storage.d_keys[0] )
            cudaFree(device_storage.d_keys[0]);
        if ( device_storage.d_keys[1] )
            cudaFree(device_storage.d_keys[1]);
    }

    void CStrRadixSortEngine::KeyValueSort(unsigned int numElem, unsigned int* h_keys, unsigned int* h_values)
    // void CStrRadixSortEngine::KeyValueSort(unsigned int numElem, unsigned long long* h_keys, unsigned int* h_values)
    {
        util::PingPongStorage<unsigned int, unsigned int> device_storage;
        //util::PingPongStorage<unsigned long long, unsigned int> device_storage;
        cudaMalloc((void**) &device_storage.d_keys[0], sizeof(unsigned int) * numElem);
        //cudaMalloc((void**) &device_storage.d_keys[0], sizeof(unsigned long long) * numElem);
        cudaMalloc((void**) &device_storage.d_values[0], sizeof(unsigned int) * numElem);

        KeysValueSort<radix_sort::LARGE_SIZE>(device_storage, numElem, h_keys, h_values);

        if ( device_storage.d_keys[0] )
            cudaFree(device_storage.d_keys[0]);
        if ( device_storage.d_keys[1] )
            cudaFree(device_storage.d_keys[1]);
        if ( device_storage.d_values[0] )
            cudaFree(device_storage.d_values[0]);
        if ( device_storage.d_values[1] )
            cudaFree(device_storage.d_values[1]);

    }
}
