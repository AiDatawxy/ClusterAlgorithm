# -*- coding: utf-8 -*
# python model_hierarchicalCluster.py


import numpy as np 
import pandas as pd 



class Node:
	''' Node class as tree node '''
	def __init__(self, layer, index, dis_lr=0, left=None, right=None, parent=None, center=None):
		self.layer = layer		# node layer
		self.index = index 		# sample index belongs to the node; -1 when the node is not a leaf
		self.dis_lr = dis_lr 	# distance of the node's left cluster and right cluster
		self.left = left 		# left child
		self.right = right 		# right child
		self.parent = parent 	# parent node
		self.center = center	# when ttype == 'center', store center of the cluster



class Cluster:
	''' Cluster class point to root of current cluster '''
	def __init__(self, name, node):
		self.name = name 
		self.node = node 



class Hierarchical():
	'''
	a simple hierarchical clustering algorithm based on data shape of DataFrame
	support self-defined distance measuring methods
	here the distances of pair points are stored in a matrix, this is inefficient which needs improving
	'''

	def __init__(self, df, num_cols, cate_cols, ttype, get_distance, gamma):
		''' initialization '''
		self.df = df
		self.num_cols = num_cols
		self.cate_cols = cate_cols
		self.ttype = ttype
		self.get_distance = get_distance
		self.gamma = gamma
		return 


	def run(self):
		''' get root '''
		root = self.calculate(self.df, self.num_cols, self.cate_cols, self.ttype, self.get_distance, self.gamma)
		return root 


	def get_cluster_points(self, root, K):
		# decode the root, and get the cluster
		k_nodes = self.get_clusterNode(root, K, self.df)
		cluster_points = []
		for i in range(K):
			cluster_points.append([])
		for i in range(K):
			self.pre_order(k_nodes[i], cluster_points[i])
		# return points index of this cluster
		return cluster_points


	def get_dis(self, samp1, samp2, dis_matrix):
		''' get distance of two points from distance matrix '''
		result = 0
		if samp1 >= samp2:
			result = dis_matrix[samp1][samp2]
		else:
			result = dis_matrix[samp2][samp1]
		return result


	def pre_order(self, node, elements):
		''' tree pre_order '''
		if node.index != -1:
			elements.append(node.index)
			return 
		if node.left:
			self.pre_order(node.left, elements)
		if node.right:
			self.pre_order(node.right, elements)


	def get_clusterDis(self, c1, c2, ttype, clusters, dis_matrix, get_distance, num_cols, cate_cols, gamma):
		''' different types of method for calculating distance of two clusters
		big, small, pair, center
		'''
		e1s = []
		e2s = []
		self.pre_order(clusters['_'.join(['c', str(c1)])].node, e1s)
		self.pre_order(clusters['_'.join(['c', str(c2)])].node, e2s)
		if ttype == 'max':
			current_dis = -1
			for e1 in e1s:
				for e2 in e2s:
					temp_dis = self.get_dis(e1, e2, dis_matrix)
					if temp_dis > current_dis:
						current_dis = temp_dis
		elif ttype == 'min':
			current_dis = float('inf')
			for e1 in e1s:
				for e2 in e2s:
					temp_dis = self.get_dis(e1, e2, dis_matrix)
					if temp_dis < current_dis:
						current_dis = temp_dis
		elif ttype == 'pair':
			current_dis = 0
			ct = 0
			for e1 in e1s:
				for e2 in e2s:
					current_dis += self.get_dis(e1, e2, dis_matrix)
					ct += 1
			current_dis /= ct
		elif ttype == 'center':
			cent1 = 0
			cent2 = 0
			for e1 in e1s:
				cent1 += df.iloc[e1][num_cols]
			for e2 in e2s:
				cent2 += df.iloc[e1][num_cols]
			current_dis = get_distance(cent1, cent2, num_cols, cate_cols, gamma)
		return current_dis


	def get_closestCluster(self, dis_clusters):
		''' get the pair points with the min distance '''
		min_key = ''
		min_dis = float('inf')
		for key, value in dis_clusters.items():
			if value < min_dis:
				min_key = key 
				min_dis = value 
		return min_key, min_dis


	def calculate(self, df, num_cols, cate_cols, ttype, get_distance, gamma):
		''' calculation process '''
		# parameter
		n = df.shape[0]		# sample numbers
		m_num = 0			# numbers of numerical variables
		m_cate = 0			# numbers of categorical variables
		if num_cols:
			m_num = len(num_cols)
		if cate_cols:
			m_cate = len(cate_cols)

		# distance matrix, store distances of pair points
		dis_matrix = []
		for i in range(n):
			dis_matrix.append([])
			for j in range(i + 1):
				dis_matrix[i].append(get_distance(df.iloc[i], df.iloc[j], num_cols, cate_cols, gamma))
		# initialize leaf node and cluster set
		clusters = {}
		for i in range(n):
			cname = '_'.join(['c', str(i)])
			clusters[cname] = Cluster(cname, Node(0, i))
		# create a dict storing distance of pair clusters
		dis_clusters = {}
		for i in range(n):
			for j in range(i):
				dis_clusters['_'.join([str(i), str(j)])] = self.get_clusterDis(i, j, ttype, clusters, dis_matrix, get_distance, num_cols, cate_cols, gamma)
		# iteration
		while len(clusters) > 1:
			# find the closest pair clusters
			closest_cluster = self.get_closestCluster(dis_clusters)
			# build new node and merge the two clusters
			i = int(closest_cluster[0].split('_')[0])
			j = int(closest_cluster[0].split('_')[1])
			left = clusters['_'.join(['c', str(i)])].node 
			right = clusters['_'.join(['c', str(j)])].node 
			temp_node = Node(max(left.layer, right.layer) + 1, -1, closest_cluster[1], left, right, None)
			left.parent = temp_node
			right.parent = temp_node
			# delete the two clusters from 'clusters'
			del clusters['_'.join(['c', str(i)])]
			del clusters['_'.join(['c', str(j)])]
			# delete disntance related to the two clusters from 'dis_clusteres'
			del_keys = []
			for key in dis_clusters.keys():
				inds = key.split('_')
				if str(i) in inds or str(j) in inds:
					del_keys.append(key)
			for del_key in del_keys:
				del dis_clusters[del_key]
			# add the new cluster into 'clusters'
			inds = []
			for key in clusters.keys():
				inds.append(int(key.split('_')[1]))
			clusters['_'.join(['c', str(i)])] = Cluster('_'.join(['c', str(i)]), temp_node)
			# add distances of the new cluster to other clusters into 'dis_clusters'
			for j in inds:
				dis_clusters['_'.join([str(i), str(j)])] = self.get_clusterDis(i, j, ttype, clusters, dis_matrix, get_distance, num_cols, cate_cols, gamma)
		root_key = list(clusters.keys())[0]
		return clusters[root_key].node


	def get_clusterNode(self, root, k, df):
		''' interpret the result '''
		n = df.shape[0]
		num = min(k, n)
		k_nodes = []
		k_nodes.append(root)
		count = 1
		while count < num:
			max_dis_lr = k_nodes[0].dis_lr 
			max_index = 0
			for i in range(1, count):
				current_dis_lr = k_nodes[i].dis_lr 
				if current_dis_lr > max_dis_lr:
					max_dis_lr = current_dis_lr
					max_index = i 
			temp_node = k_nodes[max_index]
			k_nodes[max_index] = temp_node.left
			k_nodes.append(temp_node.right)
			count += 1
		return k_nodes




# self-defined distance measuring method
def get_distance(samp1, samp2, num_cols, cate_cols, gamma):
	result = 0
	# numerical variables
	result += np.sqrt(((samp1[num_cols] - samp2[num_cols]) ** 2).sum())
	# categoridcal variables
	result += gamma * sum(samp1[cate_cols] != samp2[cate_cols])
	return(result)


if __name__ == '__main__':
	# an simple example dataframe
	df = pd.DataFrame({
		'x':[0, 0, 1, 2, 2],
		'y':[4, 0, 3, 5, 2],
		'z':['a', 'b', 'a', 'b', 'a']
	})
	num_cols = ['x', 'y'] 				# numerical variables
	cate_cols = ['z']					# categorical variables
	ttype = 'max'						# types of measuring distance between two clusters: 'max', 'min', 'pair', 'center'
	gamma = 0.5							# parameter to modify categorical variables in measuring distance

	hierarchical = Hierarchical(df, num_cols, cate_cols, ttype, get_distance, gamma)
	root = hierarchical.run()
	cluster_points_2 = hierarchical.get_cluster_points(root, 2)
	cluster_points_3 = hierarchical.get_cluster_points(root, 3)
	print('--------------------------------')
	print('2 clusters:')
	print(cluster_points_2)
	print('--------------------------------')
	print('3 clusters:')
	print(cluster_points_3)


