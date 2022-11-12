import numpy as np
import torch
import pandas as pd

i = torch.load('dqn.pth')

l = ['model.model.0.weight', 'model.model.2.weight', 'model.model.4.weight', 'model.model.6.weight', 'model.model.0.bias', 'model.model.2.bias', 'model.model.4.bias', 'model.model.6.bias']
for t in l:
    print(i[t])
j = 0
excel = pd.ExcelWriter('default.xlsx')
pd.DataFrame(i['model.model.0.weight'].numpy()).to_excel(excel, '0_weight')
pd.DataFrame(i['model.model.2.weight'].view(-1, 16).numpy()).to_excel(excel, '2_weight')
pd.DataFrame(i['model.model.4.weight'].view(-1, 16).numpy()).to_excel(excel, '4_weight')
pd.DataFrame(i['model.model.6.weight'].view(-1, 16).numpy()).to_excel(excel, '6_weight')
pd.DataFrame(i['model.model.0.bias'].numpy()).to_excel(excel, '0_bias')
pd.DataFrame(i['model.model.2.bias'].numpy()).to_excel(excel, '2_bias')
pd.DataFrame(i['model.model.4.bias'].numpy()).to_excel(excel, '4_bias')
pd.DataFrame(i['model.model.6.bias'].numpy()).to_excel(excel, '6_bias')
excel.save()
excel.close()

