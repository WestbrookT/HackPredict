import os
from nltk import word_tokenize as tok
pos = os.listdir('winners')
neg = os.listdir('losers')


total = 0
cnt = 0
for p in pos:
    cnt += 1
    with open('winners/{}'.format(p), 'r') as f:
        total += len(tok(f.read()))

for p in neg:
    cnt += 1
    with open('losers/{}'.format(p), 'r') as f:
        total += len(tok(f.read()))

print(total, cnt, total//cnt)