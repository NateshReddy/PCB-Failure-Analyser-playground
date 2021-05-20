import requests
import base64

url = 'http://172.31.10.124:4002/get-predictions'

with open("6_hold_out/00041000_temp.jpg", "rb") as f:
    img1_b64 = base64.b64encode(f.read())

with open("6_hold_out/00041000_test.jpg", "rb") as f:
    img2_b64 = base64.b64encode(f.read())

 
myobj = {'temp_img': img1_b64.decode('utf-8'), 'test_img': img2_b64.decode('utf-8')}
#print(myobj)
x = requests.post(url, json = myobj)

print(x.text)