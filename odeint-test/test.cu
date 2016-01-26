#include <iostream>

#include <boost/numeric/odeint.hpp>

#include <thrust/device_vector.h>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

using namespace std;
using namespace boost::numeric::odeint;

// state_type = vector<double>
typedef thrust::device_vector<double> state_type;
typedef runge_kutta_dopri5< state_type > stepper_type;

void rhs_circle(state_type xs, state_type &dxdt , const double t) {
  //dxdt[0] = 3.0/(2.0*t*t) + xs[0]/(2.0*t);
  dxdt[0] =  xs[1];
  dxdt[1] = -xs[0];
}

void write_cout(state_type xs , const double t) {
  cout << t << '\t';
  for (state_type::iterator iter = xs.begin(); iter != xs.end(); ++iter) {
    cout << *iter << '\t';
  }
  cout << endl;
}


int main()
{
  state_type xs(2);
  thrust::fill(xs.begin(), xs.begin() + 1, 0.0);
  thrust::fill(xs.begin() + 1, xs.begin() + 2, 1.0);

  integrate_adaptive(make_controlled(1E-12 , 1E-12 , stepper_type()),
                     rhs_circle , xs , 1.0 , 100.0, 0.1 , write_cout);
}
