# qoc: Q-learning-based Optimal Control Scheme for the Time-varying Batch Processes with Time-varying Uncertainty


## Catalog

* Data: all the data and figures for the paper. 
* env: the time-varying batch processes with time-varying uncertainties.
* algorithm: mode-based initial optimal control scheme and the Q-learning-based optimal control scheme.
* comparison_algorithm:  The comparison control scheme: PI-based indirect-type ILC from paper 'PI based indirect-type iterative learning control for batch processes with time-varying uncertainties: A 2D FM model based approach' Journal of process control,2019

## The overall Structure of the Q-learning-based optimal control scheme
![image](https://github.com/CrazyThomasLiu/qoc_for_time_varying_batch_systems/blob/master/Q_learning_based_optimal_control_scheme.jpg)

## Getting Started
* Create a python virtual environment and activate. `conda create -n qoc python=3.10` and `conda activate qoc`
* Install dependenices. `cd qoc`, `pip install -r requirement.txt` 

##  Simulation 1 : linear MIMO time-varying batch systems

* Run the following command to obtain the initial control policy and optimal control policy based on the model.
```
demo_MIMO_mbocs_final.py
```
* Run the following command to sample the data and iteratively compute the Q-learning-based optimal control policy.
```
demo_MIMO_Q_learning_final.py
```

* Run the following command to illustrate the convergence of Q-learning.
```
demo_MIMO_compare_pi_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Compare_pi_MIMO_final.jpg)

* Run the following command to compare the control performance between the initial control policy and the Q-learning-based optimal control policy.
```
demo_MIMO_test_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Q_learning_MIMO_output_final.jpg)



##  Simulation 2 : Time-varying injection molding process

* Run the following command to obtain the initial control policy and optimal control policy based on the model.
```
demo_injection_molding_mbocs_final.py
```
* Run the following command to sample the data and iteratively compute the Q-learning-based optimal control policy.
```
demo_injection_molding_Q_learning_final.py
```

* Run the following command to illustrate data sample.
```
demo_injection_molding_sample_3D_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/sample_data_3D_injection_molding_final.jpg)

* Run the following command to compare the control performance between the initial control policy and the Q-learning-based optimal control policy.
```
demo_injection_molding_test_final.py
```
![image](https://github.com/CrazyThomasLiu/Q-learning_optimal_control/blob/master/Q_learning_injection_molding_final.jpg)


