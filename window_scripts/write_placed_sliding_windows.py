import sys

step_X=sys.argv[1]
if str(step_X) == 'help':
    #              1       2         3        4      5      6       7  8 9  10
    print("args: step_X step_Y step_length X_START X_END Y_START Y_END X Y fname")
    print('python write_singular_growing_box.py 128 128 .5 300 450 500 680 700 605 retinal_growing_windows.txt')
step_X=float(sys.argv[1])
step_Y=float(sys.argv[2])
if len(sys.argv) == 11:
    step=float(sys.argv[3])
    X_START=int(sys.argv[4])
    X_END=int(sys.argv[5])
    Y_START=int(sys.argv[6])
    Y_END=int(sys.argv[7])
    X=int(sys.argv[8])
    Y=int(sys.argv[9])
    fname=sys.argv[10]
elif len(sys.argv) == 10:
    step=1
    X_START=int(sys.argv[3])
    X_END=int(sys.argv[4])
    Y_START=int(sys.argv[5])
    Y_END=int(sys.argv[6])
    X=int(sys.argv[7])
    Y=int(sys.argv[8])
    fname=sys.argv[9]
else:
    step=float(sys.argv[3])
    X_START=0#int(sys.argv[4])
    X_END=int(sys.argv[4])
    Y_START=0#int(sys.argv[6])
    Y_END=int(sys.argv[5])
    X=int(sys.argv[4])
    Y=int(sys.argv[5])
    fname=sys.argv[10]
increment_x = int(X_END-X_START)//round(step_X*step)
increment_y = int(Y_END-Y_START)//round(step_Y*step)

x_boxes=[]
y_boxes=[]

box_set = set()

X_STOP = X_END
Y_STOP = Y_END

print("step" , step)
print("x range", increment_x, 'y range', increment_y)
for i in range(int(increment_x)):# * (1. / step))):
    slide = i*step
    X_START = round(X_START + round(step_X * step) * i)
    X_END = round( X_START + step_X )#  round((step_X * step) * i +  step_X)
    if X_END > X_STOP:
        X_END = int(X_STOP)
        X_START = int((X_STOP)-step_X)
    if X_END - X_START == step_X:
        x_boxes.append("x_box "+str(X_START)+','+str(X_END))
    
for i in range(int(increment_y)):# * (1. / step))):
    slide = i/step
    Y_START = round( Y_START + round(step_Y * step) * i )
    Y_END = round( Y_START + step_Y )#round((step_Y * step) * i + step_Y)
    if Y_END > Y_STOP:
        Y_END = int(Y_STOP)
        Y_START = int((Y_STOP) - step_Y)
    if Y_END - Y_START == step_Y:
        y_boxes.append("y_box "+str(Y_START)+','+str(Y_END))

for y_box,x_box in zip(x_boxes,y_boxes):
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
