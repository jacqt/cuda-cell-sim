#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cassert>
#define N 40000

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};


int main(void)
{
    // a is [0, -1, -2, -3..]
    thrust::device_vector<int> a(N);
    thrust::sequence(a.begin(), a.end(), 0, -1);
    // b is [0*0, 1*1, .. ]
    thrust::device_vector<int> b(N);
    thrust::sequence(b.begin(), b.end(), 0, 1); //b[i] = i
    //Use template function given above for squaring
    thrust::transform(b.begin(),b.end(), b.begin(), square<int>() );
    
    thrust::device_vector<int> c(N);
    thrust::transform(a.begin(), a.end(),
                      b.begin(),
                      c.begin(),
                      thrust::plus<int>());

   
    thrust::host_vector<int> c_on_host(N);
    thrust::copy(c.begin(), c.end(), c_on_host.begin());
    for (int i=0; i<N; i++)
    {
        assert(c_on_host[i] == i*i - i);
    }
    std::cout<<"N="<<N<<"\n";
    return 0;
}
