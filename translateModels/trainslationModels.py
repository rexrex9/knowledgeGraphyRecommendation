from mxnet import nd,autograd,gluon
from datasets import gernrateDate,filePaths
import mxnet as mx
from translateModels import dataloader

def normlize(param):
    return param.norm(ord=2,axis=1,keepdims=True)

class TrainsE(gluon.nn.Block):
    def __init__(self,entity_len, relation_len, embedding_dim=200, margin=1):
        super().__init__()
        self.margin=margin
        self.entity_len=entity_len
        self.relation_len=relation_len
        self.embedding_dim = embedding_dim
        self.e = gluon.nn.Embedding(self.entity_len,embedding_dim)
        self.r = gluon.nn.Embedding(self.relation_len,embedding_dim)

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def batch_norm(self):
        for param in self.params:
            param=normlize(param)

    def predict(self,x):
        h=self.e(x[:, 0])
        r=self.r(x[:, 2])
        t=self.e(x[:, 1])
        score= h + r - t
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)

class TrainsH(gluon.nn.Block):
    def __init__(self,entity_len, relation_len, embedding_dim=200, margin=1):
        super().__init__()
        self.margin=margin
        self.entity_len=entity_len
        self.relation_len=relation_len
        self.embedding_dim = embedding_dim
        self.e = gluon.nn.Embedding(self.entity_len,embedding_dim)
        self.r = gluon.nn.Embedding(self.relation_len,embedding_dim)
        self.wr = gluon.nn.Embedding(self.relation_len,embedding_dim)


    def batch_norm(self):
        for param in self.params:
            param=normlize(param)

    def __Htransfer(self, e, wr):
        norm_wr = wr/wr.norm(ord=2,axis=1,keepdims=True)
        return e - nd.sum(e * norm_wr, 1, True) * norm_wr

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def predict(self,x):
        r_index=x[:, 2]
        h = self.e(x[:, 0])
        r = self.r(r_index)
        t = self.e(x[:, 1])
        wr = self.wr(r_index)
        score = self.__Htransfer(h,wr) + r - self.__Htransfer(t,wr)
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)


class TrainsR(gluon.nn.Block):
    def __init__(self,entity_len, relation_len, k_dim=200,r_dim=100, margin=1):
        super().__init__()
        self.margin=margin
        self.entity_len=entity_len
        self.relation_len=relation_len
        self.k_dim=k_dim
        self.r_dim=r_dim
        self.e = gluon.nn.Embedding(self.entity_len,k_dim)
        self.r = gluon.nn.Embedding(self.relation_len,r_dim)
        self.wr = gluon.nn.Embedding(self.relation_len,k_dim*r_dim)

    def batch_norm(self):
        for param in self.params:
            param=normlize(param)

    def __Rtransfer(self, e, wr):
        e=e.reshape(-1,1,self.k_dim)
        twr=wr.reshape(-1,self.k_dim,self.r_dim)
        result=nd.batch_dot(e,twr)
        result=result.reshape(-1,self.r_dim)
        return result

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def predict(self,x):
        r_index=x[:, 2]
        h = self.e(x[:, 0])
        r = self.r(r_index)
        t = self.e(x[:, 1])
        wr = self.wr(r_index)
        score = self.__Rtransfer(h,wr) + r - self.__Rtransfer(t,wr)
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)


def train(net,dataLoad,pairs,epochs=20,lr=0.01,batchSize=96):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for e in range(epochs):
        l=0
        for X in dataLoad.dataIter(pairs,batchSize):
            with autograd.record():
                loss= net.net(X)
            loss.backward()
            trainer.step(batchSize)
            net.batch_norm()
            l+=sum(loss).asscalar()
        print("Epoch {}, average loss:{}".format(e,l/len(pairs)))


if __name__ == '__main__':
    entity, relationShips, pairs = gernrateDate.getParis(filePaths.FB15K_BASE_PATH_1_1_PATH)
    net=TrainsR(len(entity),len(relationShips))
    net.collect_params().initialize(mx.init.Xavier())

    dataLoad = dataloader.Dataload(entity, relationShips)
    train(net,dataLoad,pairs)