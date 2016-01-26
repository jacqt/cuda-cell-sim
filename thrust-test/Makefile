#Note that nvcc is actually in /usr/local/cuda/bin 
# export PATH=$PATH:/usr/local/cuda/bin
#Note that there are run-time dependencies in /usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Useful
# nvidia-smi
# nvidia-smi -q

default: test_thrust_ode

test_thrust_ode: test_thrust_ode.cu
	nvcc -o $@ $<

clean:
	rm *~
