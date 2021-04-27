import sys

fraction=sys.argv[1]
X=sys.argv[2]
Y=sys.argv[3]
fname=sys.argv[4]

increment_x = int(X)//int(fraction)
increment_y = int(Y)//int(fraction)

x_boxes=[]
y_boxes=[]

for i in range(int(fraction)):
    x_boxes.append("x_box "+str(increment_x*i)+','+str(increment_x*(i+1)))
    y_boxes.append("y_box "+str(increment_y*i)+','+str(increment_y*(i+1)))
    
window_file=open(str(fname),"w+")
count = 0
total_boxes = int(fraction)**2
for x_box in x_boxes:
    for y_box in y_boxes:
        new_line = '' if count+1 == total_boxes else '\n'
        window_file.write(x_box+"\n")
        window_file.write(y_box+new_line)
        count+=1
print(count)
window_file.close()
