"""split data into training and validation sets"""
import csv
from models.defs import LID

with open('./data.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))

    #Map every language to an ID
    ID = LID

    #Write first 306 items to training set and the rest to validation set
    cnt = [0 for _ in range(6)]
    with open('./data/trainEqual.csv', 'w') as train:
        with open('./data/valEqual.csv', 'w') as val:
            for line in data:
                filepath, language = map(str.strip, line)
                id_lang = ID[language]

                if (cnt[id_lang] < 14):
                    train.write(filepath+ ',' + str(id_lang) + '\n')
                elif (cnt[id_lang] < 16):
                    val.write(filepath + ',' + str(id_lang) + '\n')
                cnt[id_lang] += 1
