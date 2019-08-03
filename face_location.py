import os

input_file_path = "./info168.dat"
add_string = "	1	0 0 168 168\n"
output_file_path = "./info168_ed.dat"
c=0
new_lines=[]
with open(input_file_path) as fd:
    content = fd.readlines()
    for line in content:
    	c+=1
    	if(c<603): 
    		continue
    	line = line.replace('\n', "")
    	new_lines.append(line+add_string)
    fd.close()
with open (output_file_path, 'w') as fd:
	for i in range(len(new_lines)):
		fd.write(new_lines[i])
