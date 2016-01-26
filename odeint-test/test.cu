#include <iostream>

#include <boost/numeric/odeint.hpp>

#include <thrust/device_vector.h>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

using namespace std;
using namespace boost::numeric::odeint;

// state_type = double
typedef thrust::device_vector<double> state_type;
typedef runge_kutta_dopri5< state_type > stepper_type;

/* we solve the simple ODE x' = 3/(2t^2) + x/(2t)
 * with initial condition x(1) = 0.
 * Analytic solution is x(t) = sqrt(t) - 1/t
 */

void rhs(state_type xs, state_type &dxdt , const double t )
{
  dxdt[0] = 3.0/(2.0*t*t) + xs[0]/(2.0*t);
  //dxdt[0] = t;
}

void write_cout( state_type x , const double t )
{
  cout << t << '\t' << x[0] << endl;
}


int main()
{
  state_type x (1);
  thrust::fill(x.begin(), x.begin() + 1, 0.0);

  integrate_adaptive( make_controlled( 1E-12 , 1E-12 , stepper_type() ) ,
      rhs , x , 1.0 , 100.0, 0.1 , write_cout );
}
