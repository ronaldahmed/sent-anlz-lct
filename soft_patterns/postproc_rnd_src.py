# clean this output
# get best conf
# run with best conf
import sys
import re
import pdb

epoch_res_re = re.compile("Epoch (?P<ep>[0-9]): (?P<acc>[0-9][.][0-9]+)")

best_acc = 0.0
best_setup = ""
setup = ""
filename = sys.argv[1]
for line in open(filename,'r'):
	line = line.strip("\n")
	if line.startswith("{"):
		setup = eval(line)
		print(line)
		
	if line.startswith("Epoch"):
		print(line)
		match = epoch_res_re.search(line)
		if match==None:
			print("Not match !!")
			pdb.set_trace()
		ep = int(match.group("ep"))
		acc = float(match.group("acc"))
		
		if acc > best_acc:
			best_acc = acc
			best_setup = setup

print("######################################################")
print(best_setup)
print(best_acc)

