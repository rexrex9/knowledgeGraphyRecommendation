import random
from datasets import filePaths
from tqdm import tqdm

def genrateRandP():
    relationShips = ['knows', 'likes', 'loves']
    people = ['p' + str(i) for i in range(100)]
    return relationShips,people

def genrateParis():
    relationShips, people = genrateRandP()
    s=set()
    for i in range(1000):
        pair=(random.choice(people),random.choice(relationShips),random.choice(people))
        s.add(pair)

    with open(filePaths.TRAINS_PATH,'w+') as f:
        for ss in s:
            f.write(str(people.index(ss[0]))+' '+str(people.index(ss[2]))+' '+str(relationShips.index(ss[1]))+' ')


def getParis(file_path):
    e,r=set(),set()
    pairs=[]
    with open(file_path,'r') as f:
        for line in tqdm(f.readlines()):
            pair=line.strip().split()
            if len(pair)!=3:
                continue
            pairs.append(pair)
            e.add(pair[0])
            e.add(pair[1])
            r.add(pair[2])
    entity=list(e)
    relationShips=list(r)
    return entity,relationShips,pairs




if __name__=='__main__':
    #genrateParis()
    people, relationShips, pairs=getParis(filePaths.TRAINS_PATH)
