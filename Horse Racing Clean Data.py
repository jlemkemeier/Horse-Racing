
"""
MNIST Example
"""

import numpy as np
import csv as csv
import pandas as pd
import sys

#read csv
with open("Data/races180509.csv", 'rb') as f:
    mycsv = csv.reader(f)
    mycsv = list(mycsv)
    
total = np.array(mycsv)

print(len(total))
print(total)




#adding finishing place
place = np.array([["place"]])
position = 0
#for x in range(1, len(total)):
date = total[0][0]
race_num = total[0][18]
for x in range(1, len(total)):
    num_horses = 0
    if total[x][0] == date and total[x][18] == race_num:
        position += 1
        place = np.vstack((place, position))
    else:
        position = 1
        date = total[x][0]
        race_num = total[x][18]
        place = np.vstack((place, position))

total = np.hstack((total, place))
print(total)

    




#sort the races
date = np.array([["ID"]])
for x in range(1, len(total)):
    #print(total[x])
    year = total[x][0][-4:]
    day = total[x][0][-6:-4]
    month = total[x][0][:-6]
    race_num = total[x][18]
    place = total[x][-1]

    if len(month) != 2:
        if len(month) == 1:
            month = '0' + month
        else:
            print("error on reading data line:", x, "data:", total[x][0])
    #print(month, day, year)
    
    date = np.vstack((date, year+month+day+race_num+place))

print(len(date))
test1 = np.hstack((date, total))
test2 = np.vstack((np.array([test1[0]]), test1[test1[:,0].argsort()]))#np.sort(test1[1:], axis=0)))
total_sorted = test2[:-1]
print(total_sorted)




#Distance: 22
#Class: 21
#Gate: 19
#Track Condition: 23
#
#Post Position: 4
#Medicine: 3
#Odds: 17
#Last 3: 16
#Horse Name: 2
#Jockey: 13
#Trainer: 14
#payout: 26,31
#place: -1
race_num_i = 18

UData = total_sorted[:,[0, 23,22,20,24,5,4,18,17, 3, 14, 15, 27, 28, 29, 30, 31, 32, -1]]
odds_i = 7
last_three_race_i = 8
horse_name_i = 9
win_i = -1
jockey_name_i = 10
trainer_name_i = 11
medicine_i = 6
pp_i = 5
pay_i = [12,13,14,15,16,17]
delete_rows = set([])
print(UData[0])





#Add a value for no medicine
for x in UData:
    if x[medicine_i] == "":
        x[medicine_i] = "NA"




#adding number of horses in racef
num_horse_race = np.array([['Number of Horses in Race']])
x = 1
total_hn = 0
total_races = 0
#for x in range(1, len(total)):
while x < len(total):
    date = total[x][0]
    race_num = total[x][18]
    num_horses = 0
    y = x
    while y<len(total) and total[y][0] == date and total[y][18] == race_num:
        y += 1
        num_horses += 1
    total_hn += num_horses
    total_races += 1
    num_horse_race = np.vstack((num_horse_race, [num_horses]))
    for z in range(y-x-1):
        num_horse_race = np.vstack((num_horse_race, [num_horses]))
    x = y

#print(win)
#print(len(win))
#print(len(num_horse_race))
print(total_hn)
average_num_horses_race = float(total_hn)/total_races
print(average_num_horses_race)
#U1Data = np.insert(UData, [2], num_horse_race, axis=1)
print(UData[0])
#print(UData)
    




for x in range(1, len(UData)):
    if not len(UData[x][pp_i]) == 1:
        #print(x[pp_i])
        delete_rows.add(x)




for x in range(1, len(UData)):
    if '' in UData[x][pay_i]:
        delete_rows.add(x)




len(delete_rows)




#converting odds to numerical

for x in range(1, len(UData)):
    #print(odds)
    #print(UData[x][7])
    odds = np.array(UData[x][odds_i].split('-'))
    #print(odds)
    if not len(odds) == 2:
        #print("delete")
        delete_rows.add(x)
    else:
        #print("not delete")
        UData[x][odds_i] = float(odds[1])/float(odds[0])
    
print(UData[0])
    
    




#converting last races
last_race = np.array([["last race"]])
second_last_race = np.array([["second last race"]])
third_last_race = np.array([["third last race"]])
#U3Data = np.array(U2Data)
for x in range(1, len(UData)):
    #print(UData[x][last_three_race_i])
    races = np.array(UData[x][last_three_race_i].split('-'))
    print(races)
    if races[0] == 'X':
        races[0] = 4
    if races[1] == 'X':
        races[1] = 4
    if races[2] == 'X':
        races[2] = 4
    last_race = np.vstack((last_race, [int(races[0])]))
    second_last_race = np.vstack((second_last_race, [int(races[1])]))
    third_last_race = np.vstack((third_last_race, [int(races[2])]))
    #print(odds)
    #print(float(odds[1])/float(odds[0]))
    #U2Data[x][7] = float(odds[1])/float(odds[0])
#U3Data =  np.insert(U3Data, [8], last_race, axis=1)
#U3Data = np.insert(U3Data, [9], second_last_race, axis=1)

#U3Data = np.insert(U3Data, [10], third_last_race, axis=1)

print(UData[0])


#U3Data = np.hstack((U3Data, last_race, second_last_race, third_last_race))
#print(U3Data)




def win_per(x):
    if float(x) == -1:
        return 1/average_num_horses_race
    return float(x)




#horse win percentage
#U4Data = np.array(U3Data)
#U4Data[13][11] = 'CHECKISINTHEMAIL'
new_horse = np.array([["new horse"]])
horse_win_per = np.array([["horse win per"]])
for x in range(1, len(UData)):
    horse_name = UData[x][horse_name_i]
    #print(horse_name)
    wins = 0
    races = 0
    for y in range(1, x):
        if UData[y][horse_name_i] == horse_name:
            #print(horse_name)
            races += 1
            #print(y)
            #print(U3Data[y][win])
            if int(UData[y][win_i]) == 1:
                #print("yes")
                wins += 1
    #print(wins, races)
    if races == 0:
        new_horse = np.vstack((new_horse, [1]))
        horse_win_per = np.vstack((horse_win_per, [-1]))
    else:
        new_horse = np.vstack((new_horse, [0]))
        horse_win_per = np.vstack((horse_win_per, [wins/races]))

horse_win_per = np.vstack(([horse_win_per[0]],[[win_per(x)] for x in horse_win_per[1:]]))






    




#jockey win percentage
#U5Data = np.array(U4Data)
#U4Data[13][11] = 'CHECKISINTHEMAIL'

jockey_win_per = np.array([["jockey win per"]])
for x in range(1, len(UData)):
    #print(jockey_name_i)
    #print(UData)
    jockey_name = UData[x][jockey_name_i]
    #print(jockey_name)
    wins = 0
    races = 0
    for y in range(1, x):
        if UData[y][jockey_name_i] == jockey_name:
            races += 1
            #print(y)
            #print(win_i)
            #print(U3Data[y][win])
            #print(UData[y][win_i])
            if int(UData[y][win_i]) == 1:
                #print("yes")
                wins += 1
    #print(wins, races)
    if races == 0:
        jockey_win_per = np.vstack((jockey_win_per, [-1]))
    else:
        #print(wins, races)
        jockey_win_per = np.vstack((jockey_win_per, [float(wins)/float(races)]))

        
jockey_win_per = np.vstack(([jockey_win_per[0]],[[win_per(x)] for x in jockey_win_per[1:]]))





#print(U5Data)









#trainer win percentage
#U6Data = np.array(U5Data)
#U4Data[13][11] = 'CHECKISINTHEMAIL'

trainer_win_per = np.array([["trainer win per"]])
for x in range(1, len(UData)):
    trainer_name = UData[x][trainer_name_i]
    #print(horse_name)
    wins = 0
    races = 0
    for y in range(1, x):
        if UData[y][trainer_name_i] == trainer_name:
            races += 1
            #print(y)
            #print(U3Data[y][win])
            if int(UData[y][win_i]) == 1:
                #print("yes")
                wins += 1
    #print(wins, races)
    if races == 0:
        trainer_win_per = np.vstack((trainer_win_per, [-1]))
    else:
        #print(wins, races)
        trainer_win_per = np.vstack((trainer_win_per, [float(wins)/float(races)]))

#print(trainer_win_per)
print(UData[0])

trainer_win_per = np.vstack(([trainer_win_per[0]],[[win_per(x)] for x in trainer_win_per[1:]]))

#print(U6Data)




UData = np.delete(UData, (8, 9, 10, 11), axis=1)



UData = np.insert(UData, [3], num_horse_race, axis=1)
UData =  np.insert(UData, [8], last_race, axis=1)
UData = np.insert(UData, [9], second_last_race, axis=1)
UData = np.insert(UData, [10], third_last_race, axis=1)
UData = np.insert(UData, [11], new_horse, axis=1)
UData = np.insert(UData, [12], horse_win_per, axis=1)
UData = np.insert(UData, [13], jockey_win_per, axis=1)
UData = np.insert(UData, [14], trainer_win_per, axis=1)






UData = np.delete(UData, list(delete_rows), axis=0)

print(UData[0])




print(UData[2])



df = pd.DataFrame(UData)
df.to_csv("Data/CleanData.csv", header = False, index = False)




