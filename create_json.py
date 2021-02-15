import json
with open('1_1500_extracted_failures.json') as f:
  data = json.load(f)

ddic = {}
# both numbers are inclusive
startnum = 151 #starting label number
endnum = 300  # ending label number
i = 1
for k in data:
  if i >= startnum and i<= endnum:
    ddic[k] = data[k]
  i+=1

with open(str(startnum)+'_'+str(endnum)+'_extracted_failures.json', 'w') as outfile:
    json.dump(ddic, outfile)
print("length of whole file", len(data))
print("length of created file", len(ddic))
