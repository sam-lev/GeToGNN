import sys

fraction=sys.argv[1]
if str(fraction) == 'help':
    print("args: fraction step X Y fname")

fraction=float(sys.argv[1])
if len(sys.argv) == 6:
    step=float(sys.argv[2])
    X=int(sys.argv[3])
    Y=int(sys.argv[4])
    fname=sys.argv[5]    
else:
    step=1
    X=int(sys.argv[2])
    Y=int(sys.argv[3])
    fname=sys.argv[4]
increment_x = int(X)//int(fraction)
increment_y = int(Y)//int(fraction)

x_boxes=[]
y_boxes=[]

box_set = set()

print("step" , step)
for i in range(int(X/((1./fraction)*X)*int(1./step))):
    slide = i*step
    X_START = round((X/fraction)*step)*i
    X_END =  round(((X/fraction)*step)*i + X/fraction)
    if X_END > X:
        X_END = int(X-1)
    x_boxes.append("x_box "+str(X_START)+','+str(X_END))
    
for i in range(int(Y/((1./fraction)*Y)*int(1./step))):
    slide = i/step
    Y_START = round((Y/fraction)*step)*i
    Y_END = round(((Y/fraction)*step)*i + Y/fraction)
    if Y_END > Y:
        Y_END = int(Y-1)
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
