
def indices_of(list:[],value)->[int]:
    indices =[]
    for i,l in enumerate(list):
        if value == l:
            indices.append(i)
    return indices

def get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]
