from mxnet import nd

#
# pw= nd.random_normal(shape=(5, 2), scale=0.01, dtype='float64')
# rw = nd.random_normal(shape=(3, 2), scale=0.01, dtype='float64')
#
# print(pw)
# print(pw[[0,1,2,0]])


#print(pw**2)
#print(sum(pw[0]**2).asscalar())


a=[6,7,8,9]
c=nd.array(a,dtype=int)
print(c)
print(nd.where(c in [6,7,8]))

#b={v:i for i,v in enumerate(a)}


print(b)