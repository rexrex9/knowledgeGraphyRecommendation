import random,copy
from mxnet import nd
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