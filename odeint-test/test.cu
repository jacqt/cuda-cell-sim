#include <iostream>

#include <boost/numeric/odeint.hpp>

#include <thrust/device_vector.h>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

using namespace std;
using namespace boost::numeric::odeint;

// state_type = vector<double>
typedef thrust::device_vector<double> state_type;
typedef runge_kutta_dopri5< state_type > stepper_type;

void rhs_circle(state_type xs, state_type &dxdts, const double t) {
  dxdts[0] =  xs[1];
  dxdts[1] = -xs[0];
}

void rhs_fitzhugh(state_type xs, state_type &dxdts, const double t) {
  float alpha = 0.2;
  float epsilon = 0.01;
  float Iapp = 0.0;
  float gamma = 0.5;
  float v = xs[0];
  float w = xs[1];
  dxdts[0] = ((-v) * (v - alpha) * (v - 1) - w + Iapp) / epsilon;
  dxdts[1] = v - (gamma * w);
}

/* format the output in a csv */
void write_cout(state_type xs , const double t) {
  cout << t;
  for (state_type::iterator iter = xs.begin(); iter != xs.end(); ++iter) {
    cout << "," << *iter;
  }
  cout << endl;
}

void fill_with_zeroes(state_type xs) {
  thrust::fill(xs.begin(), xs.end(), 0.0);
}


int main() {
  state_type xs(2); // The initial values of xs
  xs[0] = 0.64;
  xs[1] = 0.0;

  integrate_adaptive(make_controlled(1E-12 , 1E-12 , stepper_type()),
                     rhs_circle , xs , 1.0 , 100.0, 0.1 , write_cout);
}
