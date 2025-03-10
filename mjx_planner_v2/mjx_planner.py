import os

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import numpy as np
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import bernstein_coeff_order10_arbitinterval
import scipy
from functools import partial
import matplotlib.pyplot as plt 
import mujoco.mjx as mjx 
import mujoco
import time
import sys


class cem_planner():

	def __init__(self, num_dof, num_batch):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.t_fin = 4
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
		A_v_ineq, A_v = self.get_A_v()
		self.A_v_ineq = jnp.asarray(A_v_ineq) 
		self.A_v = jnp.asarray(A_v)

		A_a_ineq, A_a = self.get_A_a()
		self.A_a_ineq = jnp.asarray(A_a_ineq) 
		self.A_a = jnp.asarray(A_a)
  
		A_p_ineq, A_p = self.get_A_p()
		self.A_p_ineq = jnp.asarray(A_p_ineq) 
		self.A_p = jnp.asarray(A_p)
  
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
		self.maxiter_projection = 10
		self.maxiter_cem = 20
  
		self.l_1 = 1.0
		self.l_2 = 1.0
		self.l_3 = 1.0
		self.ellite_num = int(0.3*self.num_batch)
		

		model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
		self.model = mujoco.MjModel.from_xml_path(model_path)
		data = mujoco.MjData(self.model)
		self.model.opt.timestep = 0.02

		self.mjx_model = mjx.put_model(self.model)
		# self.mjx_data = mjx.make_data(self.model)
		self.mjx_data = mjx.put_data(self.model, data)
		init_joint_state = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])
		qpos = self.mjx_data.qpos.at[:self.num_dof].set(init_joint_state)
		self.mjx_data = self.mjx_data.replace(qpos=qpos)
		print("Timestep", self.mjx_model.opt.timestep)

		print(f'Default backend: {jax.default_backend()}')
		print('JIT-compiling the model physics step...')
		start = time.time()
		self.jit_step = jax.jit(mjx.step)
		print(f'Compilation took {time.time() - start}s.')

		object_pos = self.model.body(name="object_0").pos
		object_pos[-1] += 0.3
		self.target_pos = np.tile(object_pos, (self.num, 1))
		print(f'Target position: {self.target_pos[0]}')

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0))
	
  
	def get_A_traj(self):
	 
		A_theta = np.kron(np.identity(self.num_dof), self.P )
		A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )

		return A_theta, A_thetadot, A_thetaddot	
	
	def get_A_p(self):
	 
		A_p = np.vstack(( self.P, -self.P     ))
		A_p_ineq = np.kron(np.identity(self.num_dof), A_p )
		# A_v_ineq = np.hstack(( A_v, -A_v     ))

		return A_p_ineq, A_p
	
	def get_A_v(self):
	 
		A_v = np.vstack(( self.Pdot, -self.Pdot     ))
		A_v_ineq = np.kron(np.identity(self.num_dof), A_v )
		# A_v_ineq = np.hstack(( A_v, -A_v     ))

		return A_v_ineq, A_v

	def get_A_a(self):
	 
		A_a = np.vstack(( self.Pddot, -self.Pddot  ))
		# A_a = np.kron(np.identity(self.num_dof), self.Pddot )
  
		A_a_ineq = np.kron(np.identity(self.num_dof), A_a )

		return A_a_ineq, A_a
	
	def get_A_eq(self):

		return np.kron(np.identity(self.num_dof), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], self.Pddot[-1]    )))
	
	def get_Q_inv(self, A_eq):
		Q_inv = np.linalg.inv(np.vstack((np.hstack(( np.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_v_ineq.T, self.A_v_ineq)+self.rho_ineq*jnp.dot(self.A_a_ineq.T, self.A_a_ineq)+self.rho_ineq*jnp.dot(self.A_p_ineq.T, self.A_p_ineq), A_eq.T)  ), 
									 np.hstack((A_eq, np.zeros((np.shape(A_eq)[0], np.shape(A_eq)[0])))))))	
		return Q_inv

	@partial(jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
     
		key, subkey = random.split(key)
  	
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key
	
	@partial(jit, static_argnums=(0,))
	def compute_mean_cov(self, xi_ellite):
		xi_mean = jnp.mean(xi_ellite, axis = 0)
		xi_cov = jnp.cov(xi_ellite.T)
		return xi_mean, xi_cov


	@partial(jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):
		b_eq_term = state_term.reshape(5, self.num_dof).T
		b_eq_term = b_eq_term.reshape(self.num_dof*5)
		return b_eq_term

	@partial(jit, static_argnums=(0,))
	def compute_projection(self, v_max, a_max, p_max, lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples):
	 
		# v_max_vec = v_max*jnp.ones(( self.num_batch, self.num*self.num_dof   ))
		# a_max_vec = a_max*jnp.ones(( self.num_batch, self.num*self.num_dof   ))
		# p_max_vec = p_max*jnp.ones(( self.num_batch, self.num*self.num_dof   ))
  
		v_max_temp = jnp.hstack(( v_max*jnp.ones((self.num_batch, self.num  )),  v_max*jnp.ones((self.num_batch, self.num  ))       ))
		v_max_vec = jnp.tile(v_max_temp, (1, self.num_dof)  )

		a_max_temp = jnp.hstack(( a_max*jnp.ones((self.num_batch, self.num  )),  a_max*jnp.ones((self.num_batch, self.num  ))       ))
		a_max_vec = jnp.tile(a_max_temp, (1, self.num_dof)  )
		
		p_max_temp = jnp.hstack(( p_max*jnp.ones((self.num_batch, self.num  )),  p_max*jnp.ones((self.num_batch, self.num  ))       ))
		p_max_vec = jnp.tile(p_max_temp, (1, self.num_dof)  )
		
  
		# b_v = jnp.hstack(( v_max_vec, -v_max_vec  ))
		# b_a = jnp.hstack(( a_max_vec, -a_max_vec  ))
		# b_p = jnp.hstack(( p_max_vec, -p_max_vec  ))
  
		b_v = v_max_vec 
		b_a = a_max_vec 
		b_p = p_max_vec
		
		b_v_aug = b_v-s_v
		b_a_aug = b_a-s_a 
		b_p_aug = b_p-s_p
  
		lincost = -lamda_v-lamda_a-lamda_p-self.rho_projection*jnp.dot(self.A_projection.T, xi_samples.T).T-self.rho_ineq*jnp.dot(self.A_v_ineq.T, b_v_aug.T).T-self.rho_ineq*jnp.dot(self.A_a_ineq.T, b_a_aug.T).T-self.rho_ineq*jnp.dot(self.A_p_ineq.T, b_p_aug.T).T
		sol = jnp.dot(self.Q_inv, jnp.hstack(( -lincost, b_eq_term )).T).T
  
  
		primal_sol = sol[:, 0:self.nvar]
		s_v = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_v_ineq, primal_sol.T).T+b_v  )

		res_v = jnp.dot(self.A_v_ineq, primal_sol.T).T-b_v+s_v 

		s_a = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_a_ineq, primal_sol.T).T+b_v  )

		res_a = jnp.dot(self.A_a_ineq, primal_sol.T).T-b_a+s_a 

		s_p = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_p_ineq, primal_sol.T).T+b_p  )

		res_p = jnp.dot(self.A_p_ineq, primal_sol.T).T-b_p+s_p 
	
		lamda_v = lamda_v-self.rho_ineq*jnp.dot(self.A_v_ineq.T, res_v.T).T
		lamda_a = lamda_a-self.rho_ineq*jnp.dot(self.A_a_ineq.T, res_a.T).T
		lamda_p = lamda_p-self.rho_ineq*jnp.dot(self.A_p_ineq.T, res_p.T).T
  
		res_v_vec = jnp.linalg.norm(res_v, axis = 1)
		res_a_vec = jnp.linalg.norm(res_a, axis = 1)
		res_p_vec = jnp.linalg.norm(res_p, axis = 1)
		
		res_projection = res_v_vec+res_a_vec+res_p_vec
		
 
		return primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection

	@partial(jit, static_argnums=(0,))
	def compute_projection_filter(self, xi_samples, state_term, maxiter_projection, v_max, a_max, p_max):
	 
		b_eq_term = self.compute_boundary_vec_batch(state_term)
		s_v = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_a = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_p = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		lamda_v = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_a = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_p = jnp.zeros(( self.num_batch, self.nvar  ))
		
		for i in range(0, self.maxiter_projection):
			primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection  = self.compute_projection(v_max, a_max, p_max, lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples)
	 
		return primal_sol

	@partial(jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):
		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)
		theta = jnp.array(mjx_data.qpos[:self.num_dof])
		current_position = mjx_data.xpos[self.model.body(name="hande").id]
		eef_pos = jnp.array(current_position)
		collision = mjx_data.contact.dist<0

		return mjx_data, (theta, eef_pos, collision)

	@partial(jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot):
		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		_, out = jax.lax.scan(self.mjx_step, self.mjx_data, thetadot_single.T, length=self.num)
		theta, eef_pos, collision = out
		return theta.T.flatten(), eef_pos, collision
	
	# @partial(jit, static_argnums=(0,))
	# def step(self, batched_data, thetadot):
	# 	batched_thetadot = thetadot.reshape(self.num_batch, self.num_dof)
	# 	batch = jax.vmap(lambda mjx_data, thetadot_single: mjx_data.replace(qvel=mjx_data.qvel.at[:self.num_dof].set(thetadot_single)))(batched_data, batched_thetadot)
	# 	batched_data = self.jit_step_2(self.mjx_model, batch)
	# 	theta_eef_pose = jax.vmap(lambda mjx_data_: jnp.concatenate((jnp.array(mjx_data_.qpos[:self.num_dof]), mjx_data_.xpos[self.model.body(name="hande").id])))(batched_data)
	# 	theta, eef_pos = theta_eef_pose[:, :self.num_dof], theta_eef_pose[:, self.num_dof:]
	# 	return batched_data, (theta.flatten(), eef_pos.flatten())
	
	@partial(jit, static_argnums=(0,))
	def compute_cost_single(self, eef_pos, thetadot, collision):
		contact = jnp.sum(collision)
		w1 = 1
		# w2 = 0.005
		# w3 = 0#0.12
		w_col = 0.001

		cost_g_ = jnp.linalg.norm(eef_pos - self.target_pos, axis=1)
		cost_g = cost_g_[-1] + jnp.sum(cost_g_[:-1])*0.001

		# cost_s = jnp.sum(jnp.linalg.norm(thetadot.reshape(self.num_dof, self.num), axis=1))

		# arc_length_end = jnp.diff(eef_pos, axis = 0)
		# cost_arc = jnp.sum(jnp.linalg.norm(arc_length_end, axis = 1))

		cost = w1*cost_g+w_col*contact #+ w2*cost_s + w3*cost_arc
		return cost, cost_g_
	
	@partial(jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite
	
    

	# @partial(jit, static_argnums=(0,))
	def compute_cem(self, state_term, v_max, a_max, p_max, maxiter_projection, goal_pos):
		
		res = []
		xi_mean = jnp.zeros(self.nvar)
		xi_cov = 5*jnp.identity(self.nvar)
  
		key, subkey = random.split(self.key)
  
		time_start = time.time()
		for i in range(0, self.maxiter_cem):
		# for i in range(1):
			
			xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov ) # xi_samples are matrix of batch times (self.num_dof*self.nvar_single = self.nvar)
			xi_filtered = self.compute_projection_filter(xi_samples, state_term, maxiter_projection, v_max, a_max, p_max)
			theta_batch = jnp.dot(self.A_theta, xi_filtered.T).T 
			thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T

			# batched_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(jnp.arange(self.num_batch), self.mjx_data)
			# _, out = jax.lax.scan(self.step, batched_data, thetadot.reshape(self.num_batch*self.num_dof, self.num).T, length=self.num)
			# theta, eef_pos = out
			# theta = theta.T.reshape(self.num_batch, self.num_dof*self.num)

			theta, eef_pos, collision = self.compute_rollout_batch(thetadot)
			cost_batch, cost_g_batch = self.compute_cost_batch(eef_pos, thetadot, collision)

			xi_ellite, idx_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
			xi_mean, xi_cov = self.compute_mean_cov(xi_ellite)
			res.append(jnp.min(cost_batch))

			print(f"Iter #{i+1}: {round(time.time() - time_start, 2)}s")
			time_start = time.time()

		print(res)
		np.savetxt('output_costs1.csv',res, delimiter=",")
		

		idx_min = jnp.argmin(cost_batch)
		thetadot_best = thetadot[idx_min].reshape((self.num_dof, self.num))
		np.savetxt('best_vels.csv',thetadot_best, delimiter=",")
		np.savetxt('cost_g_best.csv',cost_g_batch[idx_min], delimiter=",")
		np.savetxt('collision.csv',collision[idx_min], delimiter=",")

		theta_single = theta[idx_min].reshape(self.num_dof, self.num).T
		plt.plot(theta_single)
		plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
		plt.savefig('best_traj.png')
  
  	
		return 0
	
def main():
	num_dof = 6
	num_batch = 500

	opt_class =  cem_planner(num_dof, num_batch)	
	theta_init = np.tile([1.5, -1.8, 1.75, -1.25, -1.6, 0], (num_batch, 1))
	thetadot_init = np.zeros((num_batch, num_dof  ))
	thetaddot_init = np.zeros((num_batch, num_dof  ))
	thetadot_fin = np.zeros((num_batch, num_dof  ))
	thetaddot_fin = np.zeros((num_batch, num_dof  ))

	state_term = np.hstack(( theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin   ))
	state_term = jnp.asarray(state_term)

	goal_pose = jnp.hstack(( 1.0, 1.0 ))

	maxiter_projection = 20
	v_max = 0.8
	a_max = 1.8
	p_max = 180*np.pi/180

	start_time = time.time()

	out = opt_class.compute_cem(state_term, v_max, a_max, p_max, maxiter_projection, goal_pose)

	print(f"Total time: {round(time.time()-start_time, 2)}s")
	
	
if __name__ == "__main__":
	main()


  	
