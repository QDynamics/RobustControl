# -*- coding: utf-8 -*-
"""
Search robust pulse, 2024 v2
"""

from qutip import *
from scipy import linalg
import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



@tf.function
def zero_constrain(tlist,T,y):
    sin_ = tf.cast(tf.math.sin(tlist/T * np.pi), tf.complex128)
    return sin_ * y

@tf.function
def create_hamiltonian_list(u_in, delta, sigmx, sigmz, tlist):
    h_list = [0.5*u_in[i]*sigmx + 0.5*delta*sigmz for i in range(len(tlist))]
    return tf.stack(h_list)

@tf.function
def expm(h_k,dt):
    return tf.linalg.expm(-1j * dt * h_k)

@tf.function
def get_U_list(hd, dt):
    U_list = [tf.constant(np.array(identity(2).full()), dtype=tf.complex128)]
    for i in range(1, len(tlist)):
        U_step = expm(hd[i],dt) @ U_list[i-1]
        U_list.append(U_step)
    return U_list

@tf.function
def get_fidelity_U(Uf, U_target):
    M = U_target @ tf.linalg.adjoint(Uf)
    MMdag = M @ tf.linalg.adjoint(M)
    M = tf.cast(M, tf.float64)
    MMdag = tf.cast(MMdag, tf.float64)
    ndim = Uf.shape[0]
    F = 1 - (1 / (ndim * (ndim + 1))) * (tf.linalg.trace(MMdag) + tf.abs(tf.linalg.trace(M)) ** 2)
    return tf.cast(F, tf.float64)

@tf.function
def z_error_distance(U_list,dt):
    sigmx = tf.constant(sigmax().full(),tf.complex128)
    sigmy = tf.constant(sigmay().full(),tf.complex128)
    sigmz = tf.constant(sigmaz().full(),tf.complex128)
    xd = [ 0.5*tf.linalg.trace(sigmx @ tf.linalg.adjoint(U_list[i]) @ sigmz @ U_list[i]) for i in range(len(U_list)) ]
    yd = [ 0.5*tf.linalg.trace(sigmy @ tf.linalg.adjoint(U_list[i]) @ sigmz @ U_list[i]) for i in range(len(U_list)) ]
    zd = [ 0.5*tf.linalg.trace(sigmz @ tf.linalg.adjoint(U_list[i]) @ sigmz @ U_list[i]) for i in range(len(U_list)) ]
    
    xd = tf.stack(xd)
    xd = tf.cast(xd,tf.float64)
    yd = tf.stack(yd)
    yd = tf.cast(yd,tf.float64)
    zd = tf.stack(zd)
    zd = tf.cast(zd,tf.float64)
    dt_ = tf.cast(dt,tf.float64)
    x = tf.stack([tf.math.reduce_sum(xd[0:i]*dt_) for i in range(len(U_list))])
    y = tf.stack([tf.math.reduce_sum(yd[0:i]*dt_) for i in range(len(U_list))])
    z = tf.stack([tf.math.reduce_sum(zd[0:i]*dt_) for i in range(len(U_list))])
    xf = tf.gather(tf.gather(x, [500]),0)
    yf = tf.gather(tf.gather(y, [500]),0)
    zf = tf.gather(tf.gather(z, [500]),0)
    r_T = tf.math.abs(xf**2 + yf**2 + zf**2)
    
    return r_T, x, y, z

@tf.function
def get_fidelity_h(hd, dt, U_target):
    U0 = tf.constant(np.array(identity(2)), dtype=tf.complex128)
    U_step = U0
    for i in range(1, len(tlist)):
        hd_complex = tf.cast(hd[i], tf.complex128)
        U_step = expm(hd[i],dt) @ U_step
        
    Uf = U_step
    M = U_target @ tf.linalg.adjoint(Uf)
    MMdag = M @ tf.linalg.adjoint(M)
    M = tf.cast(M, tf.float64)
    MMdag = tf.cast(MMdag, tf.float64)
    ndim = Uf.shape[0]
    F = 1 - (1 / (ndim * (ndim + 1))) * (tf.linalg.trace(MMdag) + abs(tf.linalg.trace(M)) ** 2)
    return tf.cast(F, tf.float64)



def optimize_pulse(pulse_init, tlist, itr, Nc, L_rate, U_target):

    dt = tf.cast(tlist[1]-tlist[0], tf.complex128)
    u1 = tf.Variable(pulse_init, name='u1', dtype=tf.float32)
    U_ideal = tf.constant(np.array(U_target.full()), dtype=tf.complex128)

    
    # optimizer = tf.keras.optimizers.Adam(learning_rate = L_rate)

    ## exponential decaying learning rate
    initial_learning_rate = L_rate    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = 100,
        decay_rate = 0.96,
        staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    

    indices = tf.reshape(tf.range(Nc+1,u1.shape[0]-Nc),shape=[u1.shape[0]-2*Nc-1,1])
    count = 0
    C_list = []  #  cost function
    F_list = []  #  fidelity
    for _ in tf.range(itr):
        with tf.GradientTape() as tape:
            u_complex1 = tf.cast(u1, tf.complex128)
            tlist = tf.cast(tlist, tf.complex128)
            
            fft_seq1 = tf.signal.fft(u_complex1)
            fft_filtered1 = fft_seq1-tf.scatter_nd(indices, fft_seq1[Nc+1:-Nc], u1.shape) # filter
            u_complex1 = tf.signal.ifft(fft_filtered1) # inverse Fourier transform
            
            u_complex1 = tf.cast(zero_constrain(tlist, T, u_complex1),tf.complex128)

            u_in = tf.cast(u_complex1,tf.complex128)
            
            sigmx = tf.constant(sigmax().full(),tf.complex128)
            sigmy = tf.constant(sigmay().full(),tf.complex128)
            sigmz = tf.constant(sigmaz().full(),tf.complex128)

            ## 
            hd = create_hamiltonian_list(u_in, 0, sigmx, sigmz, tlist)

            U_list = get_U_list(hd, dt)
            Uf = tf.gather(tf.gather(U_list, [500]),0)
            F = get_fidelity_U(Uf, U_ideal)

            r_T, x, y, z = z_error_distance(U_list,dt)
            
            C = F + r_T
            
        C_list.append((np.real(C)))
        F_list.append(np.real(F))
        if count % 10 == 0:
            print("iteration {}, cost {}".format(count,tf.abs(C)))
        count += 1
        if abs(C) < 0.000001: #
            print("break at iteration {} with cost {}".format(count,C))
            break

        dC_du1 = tape.gradient(C, u1)
        optimizer.apply_gradients(grads_and_vars=[(dC_du1, u1)])
    
    curve = [x,y,z]
    
    return np.real(u_in), C_list, F_list, curve


## initial pulse, cosine
tlist = np.linspace(0,50,501)
dt = 50/501

angle = np.pi
T = 50
p_max = angle/(0.5*T)
def f_cos(t,T_cos,p_max):
    return 0.5*p_max*(1+np.cos((t-T_cos/2)*(2*np.pi/T_cos)))

P_cos = 0.5*f_cos(tlist,T,p_max)

fig, ax = plt.subplots(figsize=(5,5*2/3))
ax.plot(tlist, P_cos/(2*np.pi), 'b', label="cos")
ax.set_xlabel('t')
ax.set_ylabel('$\\Omega/2\\pi$')
ax.legend()
plt.show()


## optimize pulse
itr = 2000
Nc = 2
L_rate = 0.1

angle = np.pi
U_target = (-1j*1/2 * angle * sigmax()).expm()

Px_i = P_cos

start_time = time.time()
P1, C_list, F_list, curve = optimize_pulse(Px_i, tlist, itr, Nc, L_rate, U_target)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The iteration took {elapsed_time} seconds to complete.")


fig, ax = plt.subplots(figsize=(5,5*2/3))
ax.plot(tlist,P1/(2*np.pi))
plt.show()
fig, ax = plt.subplots(figsize=(5,5*2/3))
ax.plot(curve[1],curve[2])
plt.show()



