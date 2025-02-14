

import numpy as np
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import bernstein_coeff_order10_arbitinterval
import scipy
from functools import partial
import matplotlib.pyplot as plt 
import os
import mujoco.mjx as mjx 
import mujoco
import time

class cem_planner():

	def __init__(self, num_dof, num_batch):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.t_fin = 5
		self.num = 50

		self.t = self.t_fin/self.num
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.A_projection = jnp.identity(self.nvar)
		self.rho_ineq = 1.0
		self.rho_projection = 1.0
		
		
		
		self.num_batch = num_batch
		# A_v_ineq, A_v = self.get_A_v()
		# self.A_v_ineq = jnp.asarray(A_v_ineq) 
		# self.A_v = jnp.asarray(A_v)

		# A_a_ineq, A_a = self.get_A_a()
		# self.A_a_ineq = jnp.asarray(A_a_ineq) 
		# self.A_a = jnp.asarray(A_a)
  
		# A_p_ineq, A_p = self.get_A_p()
		# self.A_p_ineq = jnp.asarray(A_p_ineq) 
		# self.A_p = jnp.asarray(A_p)
  
		A_eq = self.get_A_eq()
		self.A_eq = jnp.asarray(A_eq)
  
		Q_inv = self.get_Q_inv(A_eq)
		self.Q_inv = jnp.asarray(Q_inv)
  
		A_theta, A_thetadot, A_thetaddot = self.get_A_traj()
		self.A_theta = jnp.asarray(A_theta)
		self.A_thetadot = jnp.asarray(A_thetadot)
		self.A_thetaddot = jnp.asarray(A_thetaddot)
		
		self.compute_boundary_vec_batch = (vmap(self.compute_boundary_vec_single, in_axes = (0)  ))
		key = random.PRNGKey(0)

		self.key = key
		self.maxiter_projection = 1
		self.maxiter_cem = 1
  
		self.l_1 = 1.0
		self.l_2 = 1.0
		self.l_3 = 1.0
		self.ellite_num = int(0.3*self.num_batch)
		

		model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
		self.model = mujoco.MjModel.from_xml_path(model_path)
		self.model.opt.timestep = 0.1
		# self.mjx_model = mjx.load_model_from_path(model_path)
		# self.data = mujoco.MjData(self.model)
		# jax.config.update('jax_enable_x64', True)

		print(f'Default backend: {jax.default_backend()}')
		print('JIT-compiling the model physics step...')
		start = time.time()
		self.jit_step = jax.jit(mjx.step)
		print(f'Compilation took {time.time() - start}s.')
		self.mjx_model = mjx.put_model(self.model)
		print("timestep", self.mjx_model.opt.timestep)
		# mjx_data = mjx.put_data(self.model, self.data)
		  
		# self.compute_observations_batch = vmap(self.compute_observations, in_axes = (0)    )
  
		# self.set_initial_final_batch = vmap(self.set_initial_final, in_axes=(0)  )
	
  
	def get_A_traj(self):
     
		A_theta = np.kron(np.identity(self.num_dof), self.P )
		A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )

		return A_theta, A_thetadot, A_thetaddot	

	@partial(jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
     
		key, subkey = random.split(key)
  	
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key


	def get_A_eq(self):

		return np.kron(np.identity(self.num_dof), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], self.Pddot[-1]    )))

	def get_Q_inv(self, A_eq):
	#  +self.rho_ineq*torch.mm(self.A_a.T, self.A_a)
 
		Q_inv = np.linalg.inv(np.vstack((np.hstack(( np.dot(self.A_projection.T, self.A_projection), A_eq.T)  ), 
									 np.hstack((A_eq, np.zeros((np.shape(A_eq)[0], np.shape(A_eq)[0])))))))	
		return Q_inv


	@partial(jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):
	
		b_eq_term = state_term.reshape(5, self.num_dof).T

		b_eq_term = b_eq_term.reshape(self.num_dof*5)
		
		return b_eq_term

	@partial(jit, static_argnums=(0,))
	def compute_projection(self, b_eq_term,  xi_samples):
    
		lincost = -self.rho_projection*jnp.dot(self.A_projection.T, xi_samples.T).T
		sol = jnp.dot(self.Q_inv, jnp.hstack(( -lincost, b_eq_term )).T).T
    
		primal_sol = sol[:, 0:self.nvar]
		
		return primal_sol

	@partial(jit, static_argnums=(0,))
	def compute_projection_filter(self, xi_samples, state_term):
     
		b_eq_term = self.compute_boundary_vec_batch(state_term)
		
		primal_sol = self.compute_projection(b_eq_term,  xi_samples)
     
		return primal_sol

	@partial(jit, static_argnums=(0,))
	def mjx_step(self, init_state, thetadot_single):
		mjx_data = mjx.make_data(self.model)
		qpos = mjx_data.qpos.at[:self.num_dof].set(init_state[0])
		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)
		new_state = (jnp.array(mjx_data.qpos[:self.num_dof]), jnp.array(mjx_data.qvel[:self.num_dof]))
		return new_state, new_state

	@partial(jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot):
		print("thetadot inside", thetadot)
		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		print("thetadot_SINGLE inside", thetadot_single)
		# mjx_model = mjx.put_model(self.model)
		mjx_data = mjx.make_data(self.model)

		qpos = jnp.array(mjx_data.qpos[:self.num_dof])
		qvel = jnp.array(mjx_data.qvel[:self.num_dof])
		init_state = (qpos, qvel)
     
		theta = jnp.zeros((self.num_dof, self.num)) #theta=joint_states, num=steps
		theta_init = mjx_data.qpos[:self.num_dof]
		theta = theta.at[:, 0].set(theta_init)
		print("test")
		print(thetadot_single.shape)
		final_state, trajectory = jax.lax.scan(self.mjx_step, init_state, thetadot_single.T, length=self.num)
		print(trajectory)





		# for i in range(0, self.num-1):
		# 	qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single[:, i])
		# 	mjx_data = mjx_data.replace(qvel=qvel)
		# 	mjx_data = self.jit_step(mjx_model, mjx_data)
		# 	theta_vec = mjx_data.qpos[:self.num_dof]
		# 	theta = theta.at[:, i+1].set(theta_vec)
			# print(jax.debug.print("debug: {}", theta))
			# print(i)

		# print(theta.shape)
			
		return trajectory[0].T.flatten()
    

	# @partial(jit, static_argnums=(0,))
	def compute_cem(self, theta_init, state_term):
		
		res = []
		xi_mean = jnp.zeros(self.nvar)
		xi_cov = 5*jnp.identity(self.nvar)
  
		key, subkey = random.split(self.key)
  
	
		for i in range(0, self.maxiter_cem):
			
			xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov )
			## xi_samples are matrix of batch times (self.num_dof*self.nvar_single = self.nvar)
			xi_filtered = self.compute_projection_filter(xi_samples, state_term )

			thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T
			# theta = self.compute_rollout_batch(thetadot) ### mjx
			# thetadot_single = thetadot[0].reshape(self.num_dof, self.num)
			# theta = self.compute_rollout_single(thetadot[0])
			compute_rollout_batch = (vmap(self.compute_rollout_single, in_axes = (0)))
			print("thetadot outside",thetadot.shape)
			theta_batch = compute_rollout_batch(thetadot)

			print("theta_batch", theta_batch.shape)

			np.savetxt('output_1.csv',theta_batch,delimiter=",")
			
			### theta should be a matrix of batch times(self.num_dof*self.num)

   
			### thetadot is matrix of batch times(self.num_dof*self.num)
   
			# theta_single = theta_batch[0].reshape(self.num_dof, self.num)
			# plt.plot(theta.T)
			# plt.show()

			# cost_batch = self.compute_cost
  
  	
		return 0
	
def main():
	num_dof = 7
	num_batch = 100
	opt_class =  cem_planner(num_dof, num_batch)
	theta_init = np.zeros((num_batch, num_dof))
	thetadot_init = np.zeros((num_batch, num_dof  ))
	thetaddot_init = np.zeros((num_batch, num_dof  ))
	thetadot_fin = np.zeros((num_batch, num_dof  ))
	thetaddot_fin = np.zeros((num_batch, num_dof  ))
	state_term = np.hstack(( theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin   ))
	state_term = jnp.asarray(state_term)
	temp = opt_class.compute_cem(theta_init, state_term)
	
	
if __name__ == "__main__":
	main()


  	
