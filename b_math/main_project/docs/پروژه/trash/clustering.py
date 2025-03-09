

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

adj = np.array([
        [0 , 1 , 0 , 1 , 0 , 0 , 0 , 0, 0],
        [1 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0],
        [0 , 0 , 0 , 1 , 0 , 1 , 0 , 0, 0],
        [1 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0],
        [0 , 0 , 1 , 0 , 0 , 1 , 0 , 1, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 0, 0],
        [0 , 0 , 0 , 0 , 0 , 1 , 0 , 1, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 0, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 1, 0],


    ])

def show_graph(adjacency_matrix, labels=None):
    color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}
    colors = [color_map[x] for x in labels] if labels is not None else None

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, node_color=colors)
    plt.show()

#print(adj)
show_graph(adj, [1, 1, 1, 1, 2, 2, 2, 2, 2])



#def adj matris
adj = np.array([
        [0 , 1 , 0 , 1 , 0 , 0 , 0 , 0, 0],
        [1 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0],
        [0 , 0 , 0 , 1 , 0 , 1 , 0 , 0, 0],
        [1 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0],
        [0 , 0 , 1 , 0 , 0 , 1 , 0 , 1, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 0, 0],
        [0 , 0 , 0 , 0 , 0 , 1 , 0 , 1, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 0, 1],
        [0 , 0 , 0 , 0 , 1 , 0 , 1 , 1, 0],


    ])



Sum = adj.sum(axis=1)
diag_sum = np.diag(Sum)
Lapl = diag_sum - adj
print(f'Sum:\n{Sum}')
print(f'diag:\n{diag_sum}')
print(f'lapl:\n{Lapl}')



eigenvalues, eigenvectors = np.linalg.eig(adj)
#sort index
sorted_indices = np.argsort(eigenvalues)
#sorted_mat
eig_val = eigenvalues[sorted_indices]
eig_vec = eigenvectors[:, sorted_indices]

#print(f'eig_val:\n{eig_val}')
#print(f'eig_vec:\n{eig_vec}')


colors = [1 if  item < 0 else 2  for item in  eig_val ]

print(colors)

show_graph(adj , colors)


#first eig vec
vec1 = eig_vec[: ,1]

#sec eig vec
vec2 = eig_vec[: ,2]

#clustered colors
clu_colors = [ 1 if v1*v2>0 else 2 for v1,v2 in zip(vec2 , vec1)]

print(f'vec1 : {vec1} \nvec2 : {vec2}')
print(f'col_list : {clu_colors} ')



show_graph(adj , clu_colors)



adj_final = np.zeros((100, 100))
file1 = open('data.txt', 'r')
lines = file1.readlines()
del lines[0]
print(len(lines))
for l in lines:
    i, j = l.split()
    adj_final[int(i) - 1, int(j) - 1] = 1
    adj_final[int(j) - 1, int(i) - 1] = 1
print(f'adj_fin: {adj_final}')

Sum_f = adj_final.sum(axis=1)
d = np.diag(Sum_f)  # degree
Lap_f = d - adj_final
#print(Lap_f[:3])

#eig vec and vals
eig_valf , eig_vecf  = np.linalg.eigh(Lap_f)
#print(len(eig_valf))
# eig vec's
vec_f1 , vec_f2 , vec_f3 , vec_f4 = eig_vecf[ : , 1] ,eig_vecf[ : , 2] ,eig_vecf[ : , 3] ,eig_vecf[ : , 4]
clusters_2 = [1 if v1 * v2 >= 0 else 2 for v1, v2 in zip(vec_f1, vec_f2)]
#clusters_3_4 = [1 if v3 * v4 >= 0 else 2 for v3, v4 in zip(vec_f3, vec_f4)]
#clusters_1_2_3_4 = [1 if first * second == 2 else 2 for first, second in zip(clusters_2, clusters_3_4)]
another_clus = []
for v1,v2,v3,v4 in zip(vec_f1 , vec_f2 , vec_f3 , vec_f4):
  if( v1>=0  and v2<=0  and v3<=0  and v4<=0   ):
    another_clus.append(1 )

  elif( v1<=0  and v2>=0  and v3<=0  and v4<=0   ):
    another_clus.append(2 )

  elif( v1<=0  and v2<=0  and v3<=0  and v4>=0   ):
    another_clus.append( 3)

  else:
        another_clus.append(4 )




#print(clusters_2)
#print(clusters_3_4)
#print(clusters_1_2_3_4)
#print(another_clus)


show_graph(adj_final , clusters_2)
show_graph(adj_final , another_clus)