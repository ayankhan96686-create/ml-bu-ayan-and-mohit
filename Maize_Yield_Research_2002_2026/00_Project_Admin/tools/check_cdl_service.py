import urllib.request

url = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year=2020&fips=19'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        print('CropScape CDLService status:', resp.status)
        print('Response:', resp.read().decode())
except Exception as e:
    print('CropScape CDLService error:', e)

