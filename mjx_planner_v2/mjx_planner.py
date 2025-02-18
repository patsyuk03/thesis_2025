

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
		self.t_fin = 2
		self.num = 200

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
		self.maxiter_cem = 20
  
		self.l_1 = 1.0
		self.l_2 = 1.0
		self.l_3 = 1.0
		self.ellite_num = int(0.3*self.num_batch)
		

		model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
		self.model = mujoco.MjModel.from_xml_path(model_path)
		self.model.opt.timestep = 0.01

		self.mjx_model = mjx.put_model(self.model)
		print("Timestep", self.mjx_model.opt.timestep)

		print(f'Default backend: {jax.default_backend()}')
		print('JIT-compiling the model physics step...')
		start = time.time()
		self.jit_step = jax.jit(mjx.step)
		print(f'Compilation took {time.time() - start}s.')

		object_pos = self.model.body(name="object_0").pos
		object_pos[-1] += 0.2
		self.target_pos = np.tile(object_pos, (self.num, 1))
		print(f'Target position: {self.target_pos[0]}')
		  
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
	def mjx_step(self, mjx_data, thetadot_single):
		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)
		theta = jnp.array(mjx_data.qpos[:self.num_dof])
		current_position = mjx_data.xpos[self.model.body(name="hande").id]
		eef_pos = jnp.array(current_position)
		return mjx_data, (theta, eef_pos)

	@partial(jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot):
		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		mjx_data = mjx.make_data(self.model)
		init_joint_state = [1.5, -1.8, 1.75, -1.25, -1.6, 0]
		qpos = mjx_data.qpos.at[:self.num_dof].set(init_joint_state)
		mjx_data = mjx_data.replace(qpos=qpos)
		_, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
		theta, eef_pos = out
		return theta.T.flatten(), eef_pos
	
	@partial(jit, static_argnums=(0,))
	def compute_cost_single(self, eef_pos, thetadot):
		w1 = 1
		w2 = 0.001
		w3 = 1

		cost_g_ = jnp.linalg.norm(eef_pos - self.target_pos, axis=1)
		cost_g = cost_g_[-1] + jnp.sum(cost_g_[:-1])*0.001

		cost_s = jnp.sum(jnp.linalg.norm(thetadot.reshape(self.num_dof, self.num), axis=1))

		arc_length_end = jnp.diff(eef_pos, axis = 0)
		cost_arc = jnp.sum(jnp.linalg.norm(arc_length_end, axis = 1))

		cost = w1*cost_g + w2*cost_s + w3*cost_arc
		return cost, cost_g_
	
	@partial(jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite
	
	@partial(jit, static_argnums=(0,))
	def compute_mean_cov(self, xi_ellite):
		xi_mean = jnp.mean(xi_ellite, axis = 0)
		xi_cov = jnp.cov(xi_ellite.T)
		return xi_mean, xi_cov
    

	# @partial(jit, static_argnums=(0,))
	def compute_cem(self, theta_init, state_term):
		
		res = []
		xi_mean = jnp.zeros(self.nvar)
		xi_cov = 1*jnp.identity(self.nvar)
  
		key, subkey = random.split(self.key)
  
		time_start = time.time()
		for i in range(0, self.maxiter_cem):
			
			xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov ) # xi_samples are matrix of batch times (self.num_dof*self.nvar_single = self.nvar)
			xi_filtered = self.compute_projection_filter(xi_samples, state_term ) 

			thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T

			compute_rollout_batch = vmap(self.compute_rollout_single, in_axes = (0))
			theta, eef_pos = compute_rollout_batch(thetadot)

			compute_cost_batch = vmap(self.compute_cost_single, in_axes = (0))
			cost_batch, cost_g_batch = compute_cost_batch(eef_pos, thetadot)

			xi_ellite, idx_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
			xi_mean, xi_cov = self.compute_mean_cov(xi_ellite)
			res.append(jnp.min(cost_batch))

			print(f"Iter #{i+1}: {round(time.time() - time_start, 2)}s")
			time_start = time.time()

			# theta_single = theta[0].reshape(self.num_dof, self.num).T

			# np.savetxt('output_vels.csv',thetadot[0].reshape(self.num_dof, self.num).T, delimiter=",")
			# np.savetxt('output_traj.csv',theta[0].reshape(self.num_dof, self.num).T ,delimiter=",")
			
			# plt.plot(theta_single)
			# plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
			# plt.savefig('single_traj.png')
			# plt.clf()

   
			# for i in range(theta.shape[0]):
			# 	theta_single = theta[i].reshape(self.num_dof, self.num).T
			# 	plt.plot(theta_single)
			# plt.savefig('trajectories_1.png')

			# cost_batch = self.compute_cost

		print(res)
		np.savetxt('output_costs1.csv',res, delimiter=",")

		idx_min = jnp.argmin(cost_batch)
		thetadot_best = thetadot[idx_min].reshape((self.num_dof, self.num))
		np.savetxt('best_vels.csv',thetadot_best, delimiter=",")
		np.savetxt('cost_g_best.csv',cost_g_batch[idx_min], delimiter=",")

		theta_single = theta[idx_min].reshape(self.num_dof, self.num).T
		plt.plot(theta_single)
		plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
		plt.savefig('best_traj.png')
  
  	
		return 0
	
def main():
	num_dof = 6
	num_batch = 500
	opt_class =  cem_planner(num_dof, num_batch)
	# theta_init = np.zeros((num_batch, num_dof))
	theta_init = np.tile([1.5, -1.8, 1.75, -1.25, -1.6, 0], (num_batch, 1))
	thetadot_init = np.zeros((num_batch, num_dof  ))
	thetaddot_init = np.zeros((num_batch, num_dof  ))
	thetadot_fin = np.zeros((num_batch, num_dof  ))
	thetaddot_fin = np.zeros((num_batch, num_dof  ))
	state_term = np.hstack(( theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin   ))
	state_term = jnp.asarray(state_term)
	start_time = time.time()
	temp = opt_class.compute_cem(theta_init, state_term)
	print(f"Total time: {round(time.time()-start_time, 2)}s")
	
	
if __name__ == "__main__":
	main()


  	
