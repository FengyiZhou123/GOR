import Real_time_new
import utils

# dict ={}
# dict[1]=2
# print(len(dict))

tmp=dict()
tmp[1] = [1,2]
print(tmp[1][0])
print(tmp.get(1)[0])
print(tmp.get(1, [0,0])[0])