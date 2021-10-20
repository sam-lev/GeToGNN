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
increment_x = int(X)//int(step_X)
increment_y = int(Y)//int(step_Y)

x_boxes=[]
y_boxes=[]

box_set = set()

print("step" , step)
print("x range", increment_x, 'y range', increment_y)
for i in range(int(increment_x * (1. / step))):
    slide = i*step
    X_START = round(step_X * step) * i
    X_END =  round((step_X * step) * i +  step_X)
    if X_END > X:
        X_END = int(X-1)
        X_START = int((X-1)-step_X)
    if X_END - X_START == step_X:
        x_boxes.append("x_box "+str(X_START)+','+str(X_END))
    
for i in range(int(increment_y * (1. / step))):
    slide = i/step
    Y_START = round(step_Y * step) * i
    Y_END = round((step_Y * step) * i + step_Y)
    if Y_END > Y:
        Y_END = int(Y-1)
        Y_START = int((Y-1) - step_Y)
    if Y_END - Y_START == step_Y:
        y_boxes.append("y_box "+str(Y_START)+','+str(Y_END))

for x_box in x_boxes:
    for y_box in y_boxes:
        box_set.add((x_box, y_box))
        
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
