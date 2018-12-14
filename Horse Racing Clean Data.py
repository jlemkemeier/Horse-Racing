
"""
MNIST Example
"""

import numpy as np
import csv as csv
import pandas as pd
import sys

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
trainer_win_per = np.array([["trainer win per"]])
num_horse_race = np.array([['Number of Horses in Race']])
last_race = np.array([["last race"]])
second_last_race = np.array([["second last race"]])
third_last_race = np.array([["third last race"]])
new_horse = np.array([["new horse"]])
horse_win_per = np.array([["horse win per"]])
jockey_win_per = np.array([["jockey win per"]])
trainer_win_per = np.array([["trainer win per"]])
average_num_horses_race =0



def main():
    with open("Data/races180509.csv", 'rb') as f:
        mycsv = csv.reader(f)
        mycsv = list(mycsv)
        
    total = np.array(mycsv)

    print(len(total))
    print(total[0])

    total = add_finishing_place(total)
    total_sorted = sort_races(total)
    UData = extract_useful_info(total_sorted)
    add_value_for_no_medicine(UData)
    adding_num_horses_in_race(UData, total)
    converting_odds_to_numerical(UData)
    converting_previous_races(UData)
    calculating_horse_win_percentage(UData)
    calculating_jockey_win_percentage(UData)
    calculating_trainer_win_percentage(UData)

    UData = add_and_delete_rows(UData)
    save_data(UData)




#adding finishing place
def add_finishing_place(total):
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
    print("print what is happening", total[0])

    return np.hstack((total, place))


#sort the races
def sort_races(total):
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

    test1 = np.hstack((date, total))
    test2 = np.vstack((np.array([test1[0]]), test1[test1[:,0].argsort()]))#np.sort(test1[1:], axis=0)))
    total_sorted = test2[:-1]

    return total_sorted




def extract_useful_info(total_sorted):
    print("GAWDDAMN IT TELL ME WHAT IS HAPPENING", total_sorted[0])
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
    return UData



#Add a value for no medicine
def add_value_for_no_medicine(UData):
    for x in UData:
        if x[medicine_i] == "":
            x[medicine_i] = "NA"




#adding number of horses in race
def adding_num_horses_in_race(UData, total):
    global num_horse_race
    global average_num_horses_race
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
    average_num_horses_race = float(total_hn)/total_races

    for x in range(1, len(UData)):
        if not len(UData[x][pp_i]) == 1:
            delete_rows.add(x)


    for x in range(1, len(UData)):
        if '' in UData[x][pay_i]:
            delete_rows.add(x)








#converting odds to numerical
def converting_odds_to_numerical(UData):
    for x in range(1, len(UData)):
        odds = np.array(UData[x][odds_i].split('-'))
        if not len(odds) == 2:
            delete_rows.add(x)
        else:
            UData[x][odds_i] = float(odds[1])/float(odds[0])
        
        
    




#converting last races
def converting_previous_races(UData):
    global last_race, second_last_race, third_last_race
    for x in range(1, len(UData)):
        races = np.array(UData[x][last_three_race_i].split('-'))
        if races[0] == 'X':
            races[0] = 4
        if races[1] == 'X':
            races[1] = 4
        if races[2] == 'X':
            races[2] = 4
        last_race = np.vstack((last_race, [int(races[0])]))
        second_last_race = np.vstack((second_last_race, [int(races[1])]))
        third_last_race = np.vstack((third_last_race, [int(races[2])]))




def win_per(x):
    #print(x)
    if float(x[0]) == -1:
        return 1/average_num_horses_race
    return float(x[0])




#horse win percentage
def calculating_horse_win_percentage(UData):
    global horse_win_per
    global new_horse
    print("start calculations")
    print("win_i", win_i)
    print("column titles", UData[0])
    print(UData[:,win_i])
    for x in range(1, len(UData)):
        horse_name = UData[x][horse_name_i]
        wins = 0
        races = 0
        for y in range(1, x):
            if UData[y][horse_name_i] == horse_name:
                races += 1
                if int(UData[y][win_i]) == 1:
                    wins += 1
        if races == 0:
            new_horse = np.vstack((new_horse, [1]))
            horse_win_per = np.vstack((horse_win_per, [-1]))
        else:
            new_horse = np.vstack((new_horse, [0]))
            horse_win_per = np.vstack((horse_win_per, [wins/float(races)]))
        if x == 1:
            print(horse_win_per[x])

    horse_win_per = np.vstack(([horse_win_per[0]],[[win_per(x)] for x in horse_win_per[1:]]))
   #print("Here1", horse_win_per)





    




#jockey win percentage
def calculating_jockey_win_percentage(UData):
    global jockey_win_per
    for x in range(1, len(UData)):
        jockey_name = UData[x][jockey_name_i]
        wins = 0
        races = 0
        for y in range(1, x):
            if UData[y][jockey_name_i] == jockey_name:
                races += 1
                if int(UData[y][win_i]) == 1:
                    wins += 1
        if races == 0:
            jockey_win_per = np.vstack((jockey_win_per, [-1]))
        else:
            jockey_win_per = np.vstack((jockey_win_per, [float(wins)/float(races)]))

            
    jockey_win_per = np.vstack(([jockey_win_per[0]],[[win_per(x)] for x in jockey_win_per[1:]]))
    #print("HERE2", jockey_win_per)











#trainer win percentage
def calculating_trainer_win_percentage(UData):
    global trainer_win_per
    for x in range(1, len(UData)):
        trainer_name = UData[x][trainer_name_i]
        wins = 0
        races = 0
        for y in range(1, x):
            if UData[y][trainer_name_i] == trainer_name:
                races += 1
                if int(UData[y][win_i]) == 1:
                    wins += 1
        if races == 0:
            trainer_win_per = np.vstack((trainer_win_per, [-1]))
        else:
            trainer_win_per = np.vstack((trainer_win_per, [float(wins)/float(races)]))

    trainer_win_per = np.vstack(([trainer_win_per[0]],[[win_per(x)] for x in trainer_win_per[1:]]))




def add_and_delete_rows(UData):
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
    return UData

def save_data(UData):
    print(UData[0])
    df = pd.DataFrame(UData)
    df.to_csv("Data/CleanData.csv", header = False, index = False)


main()



