#include <iostream>

#include <boost/numeric/odeint.hpp>

#include <thrust/device_vector.h>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
#include <math.h>

using namespace std;
using namespace boost::numeric::odeint;

// state_type = vector<double>
typedef thrust::device_vector<float> state_type;
// typedef runge_kutta_dopri5< state_type > stepper_type;
typedef runge_kutta4< state_type > stepper_type;


void rhs_circle(state_type xs, state_type &dxdts, const float t) {
  dxdts[0] =  xs[1];
  dxdts[1] = -xs[0];
}

void rhs_fitzhugh(state_type xs, state_type &dxdts, const float t) {
  float alpha = 0.2;
  float epsilon = 0.01;
  float Iapp = 0.0;
  float gamma = 0.5;
  float v = xs[0];
  float w = xs[1];
  dxdts[0] = ((-v) * (v - alpha) * (v - 1) - w + Iapp) / epsilon;
  dxdts[1] = v - (gamma * w);
}


struct output_observer {

  float* cumulative_error_;
  output_observer(float *error) : cumulative_error_(error) { }
  void operator()(const state_type xs, const float t) {
    float e_x = cos(t - 1);
    float e_y = - sin(t - 1);
    float error = (xs[0] - e_x) * (xs[0] - e_x) + (xs[1] - e_y) * (xs[1] - e_y);
    *cumulative_error_ += error;
  }
};

// Runs the euler ode solver
float get_euler_error(double timestep) {
  state_type xs(2); // The initial values of xs
  xs[0] = 1.0;
  xs[1] = 0.0;
  float error = 0.0;
  output_observer observer = output_observer(&error);
  integrate_const(euler< state_type >(),
                  rhs_circle,
                  xs, 1.0, 20.0, timestep,
                  observer);
  return sqrt(error);
}

// Runs the runge_katta4 ode solver
float get_const_error(double timestep) {
  state_type xs(2); // The initial values of xs
  xs[0] = 1.0;
  xs[1] = 0.0;
  float error = 0.0;
  output_observer observer = output_observer(&error);
  integrate_const(runge_kutta4< state_type >(),
                  rhs_circle,
                  xs, 1.0, 20.0, timestep,
                  observer);
  return sqrt(error);
}

// Runs the runge_kutta_dopri5 ode solver
float get_adaptive_error(double timestep) {
  state_type xs(2); // The initial values of xs
  xs[0] = 1.0;
  xs[1] = 0.0;
  float error = 0.0;
  output_observer observer = output_observer(&error);
  integrate_adaptive(make_controlled(1E-12, 1E-12, runge_kutta_dopri5< state_type >()),
                     rhs_circle,
                     xs, 1.0, 20.0, timestep,
                     observer);
  return sqrt(error);
}

// Runs the ode solver over a set of trials
void run_trials() {
  for (double timestep = 0.0001; timestep < 1; timestep *= sqrt(10)) {
    float euler_error  =  get_euler_error(timestep);
    float adaptive_error =  get_adaptive_error(timestep);
    float const_error = get_const_error(timestep);

    cout << timestep << ", " << euler_error  << ", " << const_error << ", " << adaptive_error << endl;
  }
}

int main() {
  run_trials();
}
