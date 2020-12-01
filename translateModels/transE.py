from mxnet import nd,autograd
from datasets import gernrateDate,filePaths
import random
import copy


class TrainsE():
    def __init__(self,entity_len, relation_len, embedding_dim=200, margin=1,**kwargs):
        super(TrainsE, self).__init__(**kwargs)
        self.margin=margin
        self.entity_len=entity_len
        self.relation_len=relation_len
        self.embedding_dim = embedding_dim

    def __batch_norm(self):
        for param in self.params:
            param/= nd.sum(param**2, axis=1, keepdims=True)**0.5

    def __init_adam_states(self):
        self.adam_t = 1
        self.adam_states=[]
        for param in self.params:
            self.adam_states.append((nd.zeros(param.shape),nd.zeros(param.shape)))

    def initialize(self):
        self.ew = nd.random_normal(shape=(self.entity_len, self.embedding_dim),scale=6/(self.embedding_dim**2))
        self.rw = nd.random_normal(shape=(self.relation_len, self.embedding_dim), scale=6/(self.embedding_dim**2))
        self.params = [self.ew, self.rw]
        self.__batch_norm()
        self.__init_adam_states()
        for param in self.params:
            param.attach_grad()

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def predict(self,x):
        a= self.ew[x[:, 0]] + self.rw[x[:, 2]] - self.ew[x[:, 1]]
        return nd.sum(a**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt),y_correct,y_corrupt

    def SGD(self,lr):
        for param in self.params:
            param[:] = param - lr * param.grad
        self.__batch_norm()

    def adam(self,lr):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        for p, (v, s) in zip(self.params, self.adam_states):
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * p.grad.square()
            v_bias_corr = v / (1 - beta1 ** self.adam_t)
            s_bias_corr = s / (1 - beta2 ** self.adam_t)
            p[:] -= lr * v_bias_corr / (s_bias_corr.sqrt() + eps)
        self.adam_t += 1
        self.__batch_norm()

class Dataload():

    def __init__(self,entity_list,relation_list):
        self.entity_list=entity_list
        self.relation_list=relation_list
        self.entity2id = {v: i for i, v in enumerate(entity_list)}
        self.r2id = {v: i for i, v in enumerate(relation_list)}

    def __corrupt(self,datasets):
        corrupted_datasets = []
        for triple in datasets:
            corrupted_triple = copy.deepcopy(triple)
            seed = random.random()
            if seed > 0.5:# 替换head
                rand_head = triple[0]
                while rand_head == triple[0]:
                    rand_head = random.sample(self.entity_list, 1)[0]
                corrupted_triple[0] = rand_head
            else:# 替换tail
                rand_tail = triple[1]
                while rand_tail == triple[1]:
                    rand_tail = random.sample(self.entity_list, 1)[0]
                corrupted_triple[1] = rand_tail
            corrupted_datasets.append(corrupted_triple)
        return corrupted_datasets


    def __getIndexes(self,dataset):
        r_dataset=[[self.entity2id[d[0]], self.entity2id[d[1]], self.r2id[d[2]]] for d in dataset]
        return r_dataset

    def dataIter(self,pairs,batchSize):
        for i in range(len(pairs)//batchSize):
            dataset = random.sample(pairs, batchSize)
            corrupted_datasets=self.__corrupt(dataset)
            yield nd.array(self.__getIndexes(dataset)),nd.array(self.__getIndexes(corrupted_datasets))

def train(entity,relationShips,pairs,epochs=100,lr=0.01,batchSize=100):
    net=TrainsE(len(entity),len(relationShips))
    net.initialize()
    dataLoad=Dataload(entity,relationShips)


    for e in range(epochs):
        l,lc,ly=0,0,0
        for X in dataLoad.dataIter(pairs,batchSize):
            with autograd.record():
                loss,yc,yy= net.net(X)
            loss.backward()
            net.adam(lr)
            l+=sum(loss).asscalar()
            lc+=sum(yc).asscalar()
            ly += sum(yy).asscalar()
        print("Epoch {}, average loss:{},{},{}".format(e, l/1000,lc/1000,ly/1000))

    print(net.ew[:5])

if __name__ == '__main__':

    entity, relationShips, pairs = gernrateDate.getParis(filePaths.FB15K_BASE_PATH_1_1_PATH)
    train(entity, relationShips, pairs)
