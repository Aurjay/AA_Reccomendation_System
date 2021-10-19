
import os
from deep_translator import GoogleTranslator
x=[]
#Here you should replace with your path for file.
path_dir = "C:\\Users\\Deepak\\Desktop\\azcopy_windows_amd64_10.12.2\\tum\\TUMAI_dataset\\pdp_dataset\\KellerSports"
Keller_sports = os.listdir(path_dir)
#Here the product description of all different folders are accessed.
path_list = []
for y in Keller_sports:
    path_list.append("{}\\{}\\{}".format(path_dir, y, 'product_description.txt'))
for z in path_list:
    with open(z,'r',encoding = 'utf8') as f:
        flat_list = [word for line in f for word in line.split()]
#Google translator has a 5000 characters limit.
    translated = GoogleTranslator(source='de', target='en').translate_batch(flat_list)
    translatedd = " ".join(translated)
    h = open(z, 'w', encoding='utf8')
    h.write(translatedd)
print("Translated")
