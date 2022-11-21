This repository contains the data of robust control pulses and code demo in the paper 
*Universal Robust Quantum Gates by Geometric Correspondence*.
Paper link: [arXiv:2210.14521](https://arxiv.org/abs/2210.14521)

The csv data of the robust control pulses are in the file RCP_data, where the pulses are defined in the time domain tlist = np.linspace(0,50,501). 
The units of the pulse amplitude and time are GHz and ns respectively.

The code RCP_robustness.ipynb demonstrates the robustness of the quantum gate generated by each robust control pulse.

The code Pulse_construction_demo.ipynb is a simple demonstration of the pulse construction protocol to search for the robust control pulses.

