import sys

step_X=sys.argv[1]
if str(step_X) == 'help':
    #              1       2         3        4      5      6       7     8    9    10    11    12
    print("args: step_X step_Y step_length X_START X_END Y_START Y_END X_MIN Y_MIN X_MAX Y_MAX fname")
    print('python write_singular_growing_box.py 128 128 .5 300 450 500 680 700 605 retinal_growing_windows.txt')
step_X=float(sys.argv[1])
step_Y=float(sys.argv[2])
if len(sys.argv) == 13:
    step=float(sys.argv[3])
    X_START=int(sys.argv[4])
    X_END=int(sys.argv[5])
    Y_START=int(sys.argv[6])
    Y_END=int(sys.argv[7])
    X_MIN=int(sys.argv[8])
    X_MAX=int(sys.argv[9])
    Y_MIN = int(sys.argv[10])
    Y_MAX = int(sys.argv[11])
    fname=sys.argv[12]
elif len(sys.argv) == 9:
    step = float(sys.argv[3])
    X_START = int(sys.argv[4])
    X_END = int(sys.argv[5])
    Y_START = int(sys.argv[6])
    Y_END = int(sys.argv[7])
    X_MIN = X_START
    X_MAX = X_END#int(sys.argv[9])
    Y_MIN = Y_START#int(sys.argv[10])
    Y_MAX = Y_END#int(sys.argv[11])
    fname = sys.argv[8]
else:
    step=float(sys.argv[3])
    X_START=0#int(sys.argv[4])
    X_END=int(sys.argv[4])
    Y_START=0#int(sys.argv[6])
    Y_END=int(sys.argv[5])
    X=int(sys.argv[4])
    Y=int(sys.argv[5])
    fname=sys.argv[10]
increment_x = (X_END-X_START)/(step_X*step)
increment_y = (Y_END-Y_START)/(step_Y*step)

entire_boxes_x = (X_MAX - X_MIN)/(step_X*step)
entire_boxes_y = (Y_MAX - Y_MIN)/(step_Y*step)

x_boxes=[]
y_boxes=[]

box_set = []

box_set_y = []
box_set_x = []

print("step" , step)
print("x range", increment_x, 'y range', increment_y)
ends = 0
begun_y = True
begun_x = True

Y_STOP = Y_END
X_STOP = X_END
Y_INIT = Y_START
X_INIT = X_START

for m in range(int(entire_boxes_y)):
    for n in range(int(entire_boxes_x)):#
        last_stop_y = Y_STOP
        last_stop_x = X_STOP
        X_STOP = X_END if begun_x else round(X_STOP + (step_X))
        X_INIT = X_INIT if begun_x else round(X_INIT - (step_X))
        if X_INIT < X_MIN:
            X_INIT = X_MIN
        if X_STOP > X_MAX:
            X_STOP = X_MAX

        Y_STOP = Y_END if begun_y else round(Y_STOP + step_Y)
        Y_INIT = Y_INIT if begun_y else round(Y_INIT - step_Y)
        if Y_INIT < Y_MIN:
            Y_INIT = Y_MIN
        if Y_STOP > Y_MAX:
            Y_STOP = Y_MAX


        for i in range(int(increment_y+1)):
            for j in range(int(increment_x+1)):# * (1. / step))):

                X_START = round(X_START + round(step_X * step * i))
                X_END = round( X_START + step_X )#  round((step_X * step) * i +  step_X)
                #for i in range(int(increment_y)):# * (1. / step))):
                Y_START = round( Y_START + round(step_Y * step * j ))
                Y_END = round( Y_START + step_Y )#round((step_Y * step) * i + step_Y)
                #if (Y_END <= Y_STOP and Y_END-Y_START == step_Y) and ( X_END <= X_STOP and (X_END-X_START == step_X)):

                y_box = "y_box "+str(int(Y_START))+','+str(int(Y_END))
                if Y_END > Y_STOP:
                        y_box = "y_box " + str(int(Y_STOP - step_Y)) + ',' + str(int(Y_STOP))
                x_box = "x_box "+str(int(X_START))+','+str(int(X_END))
                if X_END > X_STOP:
                        x_box = "x_box " + str(int(X_STOP - step_X)) + ',' + str(int(X_STOP))
                if (x_box, y_box) not in box_set:
                    box_set.append((x_box, y_box))
                if X_END > X_STOP:
                    x_box = "x_box "+str(int(X_STOP-step_X))+','+str(int(X_STOP))

                    y_box = "y_box " + str(int(Y_START)) + ',' + str(int(Y_END))
                    if Y_END > Y_STOP:
                        y_box = "y_box " + str(int(Y_STOP - step_Y)) + ',' + str(int(Y_STOP))

                    if (x_box, y_box) not in box_set:
                        box_set.append((x_box, y_box))
                    X_START = X_INIT
                if Y_END > Y_STOP:
                    x_box = "x_box " + str(int(X_START)) + ',' + str(int(X_END))
                    if X_END > X_STOP:
                        x_box = "x_box " + str(int(X_STOP - step_X)) + ',' + str(int(X_STOP))

                    y_box = "y_box " + str(int(Y_STOP - step_Y)) + ',' + str(int(Y_STOP))

                    if (x_box, y_box) not in box_set:
                        box_set.append((x_box, y_box))
                    Y_START = Y_INIT
                begun_y = False
                begun_x = False


outer = x_boxes if len(x_boxes) < len(y_boxes) else y_boxes
inner = x_boxes if len(x_boxes) > len(y_boxes) else y_boxes
#for x_box,y_box in zip(x_boxes, y_boxes):
#    box_set.add((x_box, y_box))
# for x_box in x_boxes:
#     for y_box in y_boxes:
#         box_set.add((x_box, y_box))
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
