import sys
from data_ops.utils import linear_idx_from_coord

step_X=sys.argv[1]
if str(step_X) == 'help':
    print("args: step_X step_Y step_length X Y fname")

step_X=float(sys.argv[1])
step_Y=float(sys.argv[2])
if len(sys.argv) == 7:
    step=float(sys.argv[3])
    X=int(sys.argv[4])
    Y=int(sys.argv[5])
    fname=sys.argv[6]
else:
    step=1
    X=int(sys.argv[3])
    Y=int(sys.argv[4])
    fname=sys.argv[5]

growth_x = round(step_X * step)
growth_y = round(step_Y * step)
increment_x = (int(X-step_X)//int(growth_x)) + 1
increment_y = (int(Y-step_Y)//int(growth_y)) + 1

x_boxes=[]
y_boxes=[]

box_set = set()

print("step" , step)
print("growth ", growth_x, ' y: ', growth_y)
print("x range", increment_x, 'y range', increment_y)

X_bound = step_X
Y_bound = step_Y
for i_x, i_y in zip(range(increment_x), range(increment_y)):
    slide = i_x*step
    X_START = 0#round(step_X * step) * i_x
    X_END =  X_bound#round((step_X * step) * i_x +  step_X)
    X_bound += growth_x
    if X_END > X:
        X_END = int(X-1)
        X_START = int((X-1)-step_X)
    #x_boxes.append(
    x_box = "x_box "+str(int(X_START))+','+str(int(X_END))

    slide = i_y/step
    Y_START = 0#round(step_Y * step) * i_y
    Y_END = Y_bound#round((step_Y * step) * i_y + step_Y)
    Y_bound += growth_y
    if Y_END > Y:
        Y_END = int(Y-1)
        Y_START = int((Y-1) - step_Y)
    #y_boxes.append(
    y_box = "y_box "+str(int(Y_START))+','+str(int(Y_END))

    box_set.add((x_box, y_box))
box_set = sorted(box_set)
# for x_box in x_boxes:
#     for y_box in y_boxes:
#         box_set.add((x_box, y_box))
#
window_file=open(str(fname),"w+")
count = 0
total_boxes = len(box_set)#len(x_boxes)*len(y_boxes)
for box_pair in box_set:
    x_box = box_pair[0]
    y_box = box_pair[1]
    new_line = '' if count+1 == total_boxes else '\n'
    window_file.write(x_box+"\n")
    window_file.write(y_box+new_line)
    if count < 3:
        print(x_box)
        print(y_box)
    count+=1
print(count)
window_file.close()
