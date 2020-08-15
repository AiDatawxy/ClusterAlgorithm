# -*- coding: utf-8 -*
# python model_kPrototype.py

import numpy as np 
import pandas as pd 



class KPrototpye():
	''' A simple k-prototype clustering algorithm based on data shape of DataFrame '''


	def __init__(self, df, num_cols, cate_cols, K, gamma, epsilon, calcu_func):
		''' 初始化 '''
		self.df = df 
		self.num_cols = num_cols
		self.cate_cols = cate_cols
		self.K = K
		self.gamma = gamma
		self.epsilon = epsilon
		self.calcu_func = calcu_func
		n = df.shape[0]
		self.dis_matrix = np.zeros(n * K).reshape((n, K))  # distances of point to center
		return 
	    

	def run(self):
		''' calculation process '''
		if self.calcu_func == 'epsilon':
			categories, centroids, total_loss = self.calculate(self.df, self.num_cols, self.cate_cols, self.K, self.gamma, self.epsilon, self.calculate_epsilon)
		elif self.calcu_func == 'stable':
			categories, centroids, total_loss = self.calculate(self.df, self.num_cols, self.cate_cols, self.K, self.gamma, self.epsilon, self.calculate_stable)
		else:
			return None 
		return categories, centroids, total_loss


	def get_distance(self, samp1, samp2, num_cols, cate_cols, gamma):
		''' function of measuring distance '''
		result = 0
		# Euclidean distance on numeric variables
		result += np.sqrt(((samp1[num_cols] - samp2[num_cols]) ** 2).sum())
		# Similarity on categorical variables
		result += gamma * sum(samp1[cate_cols] != samp2[cate_cols])
		return(result)


	def get_allDistances(self, samples, centroids, num_cols, cate_cols, n, k, gamma, dis_matrix):
		''' calculate distance from point to center '''
		for i in range(n):
		    for j in range(k):
		        dis_matrix[i][j] = self.get_distance(samples.iloc[i], centroids[j], num_cols, cate_cols, gamma)


	def update_centroids(self, samples, centroids, num_cols, cate_cols, n, K, categories):
		''' adjust clustering center '''
		for cate in range(K):
		    if num_cols:
		        centroids[cate][num_cols] = samples[categories == cate][num_cols].mean()
		    if cate_cols:
		        for cate_col in cate_cols:
		            centroids[cate][cate_col] = pd.value_counts(samples[categories == cate][cate_col]).index[0]


	def get_totalLoss(self, categories, k):
		''' total loss '''
		loss = 0
		for cate in range(k):
		    loss += self.dis_matrix[categories == cate][:, cate].sum()
		return loss


	def calculate_epsilon(self, samples, centroids, num_cols, cate_cols, n, K, gamma, epsilon):
		''' clustering 1: stop iterating as the difference between losses of two iteration is less than the threshold epsilon   '''
		categories = np.random.randint(0, K, size=n)
		total_loss = -1
		current_loss = float('inf')
		while total_loss == -1 or total_loss - current_loss > epsilon:
		    total_loss = current_loss
		    # distance matrix
		    self.get_allDistances(samples, centroids, num_cols, cate_cols, n, K, gamma, self.dis_matrix)
		    # current total loss
		    current_loss = self.get_totalLoss(categories, K)
		    # seperate points into different clusters
		    categories = self.dis_matrix.argmin(axis=1)
		    # adjust center of clusters
		    self.update_centroids(samples, centroids, num_cols, cate_cols, n, K, categories)
		return categories, centroids, current_loss


	def calculate_stable(self, samples, centroids, num_cols, cate_cols, n, K, gamma, epsilon):
		''' clustering 2: stop iterating as points in each clusters do not change '''
		categories = np.random.randint(0, K, size=n)
		while True:
		    # distance matrix
		    self.get_allDistances(samples, centroids, num_cols, cate_cols, n, K, gamma, self.dis_matrix)
		    # current total loss
		    total_loss = self.get_totalLoss(categories, K)
		    # seperate points into different clusters
		    temp_categories = self.dis_matrix.argmin(axis=1)
		    if (categories != temp_categories).sum() == 0:
		        break
		    categories = temp_categories.copy()
		    # adjust center of clusters
		    self.update_centroids(samples, centroids, num_cols, cate_cols, n, K, categories)
		return categories, centroids, total_loss


	def calculate(self, df, num_cols, cate_cols, K, gamma, epsilonm, calculate_func):
		''' choose the initial center points randomly '''	
		n = df.shape[0]
		num = min(n, K)
		centroids_index = np.random.choice(n, num, replace=False)
		centroids = []
		for i in centroids_index:
		    cols = []
		    if num_cols:
		        cols += num_cols
		    if cate_cols:
		        cols += cate_cols
		    centroids.append(df.iloc[i][cols].copy())
		return calculate_func(df, centroids, num_cols, cate_cols, n, K, gamma, epsilon)





if __name__ == '__main__':
	# a simple example dataframe
	df = pd.DataFrame({								
		'c1':['a', 'b', 'c', 'a', 'c', 'b', 'a'],
		'c2':['B', 'C', 'A', 'C', 'B', 'A', 'C'],
		'r1':[1, 3, 5, 7, 9, 11, 13],
		'r2':[2, 4, 6, 8, 10, 12, 14]
	})
	num_cols = ['r1', 'r2']							# numerical variables
	cate_cols = ['c1', 'c2']						# categorical variables
	K = 3											# clustering numbers
	gamma = 0.5										# weight parameter to modify categorical variables in calculating distnace of two points
	epsilon = 0.01									# arugument in calculate_epsilon method
	calcu_func = 'stable'							# 'stable': calculate_stable method; 'epsilon': calculate_epsilon method

	# categories numbesrs: {0, 1, 2}
	# centroids are center of the three clusters
	# total_loss 
	k_prototype = KPrototpye(df, num_cols, cate_cols, K, gamma, epsilon, calcu_func)
	categories, centroids, total_loss = k_prototype.run()
	print('--------------------------------')
	print('category nums:')
	print(pd.value_counts(categories))
	print('--------------------------------')
	print('total_loss:')
	print(total_loss)

	# appendix：k means in sklearn
	# from sklearn import cluster
	# df1 = df[['r1', 'r2']]
	# km = cluster.KMeans(init='k-means++', n_clusters=K, random_state=77)
	# result = km.fit_predict(df1)
	# result


