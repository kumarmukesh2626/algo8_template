#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:54:21 2022

@author: mukesh
"""
import ConnectDb
from ConnectDb import postgreSql
import psycopg2
from ConnectDb import sql,push_into_db
import datetime
import pandas as pd

import ast
import datetime
import pandas as pd
from datetime import timedelta

db = sql()
#db_conn = db.connect_with_sql("admin", "Dev123456", "hzl.cgwyhisjxon9.ap-south-1.rds.amazonaws.com", "3306", "hzl")
db_conn = db.connect_with_sql("root", "root123", "localhost", "3306", "ivision")
print(type(db_conn))


def getShift():
        hour = pd.Timestamp.now(tz = 'Asia/Kolkata').hour
        if hour >= 6 and hour < 14:
            return 'A'
        elif hour >= 14 and hour < 22:
            return 'B'
        return 'C'
    

def check_offset_lunch():
    shift = getShift()
    print(shift)
    old_shift = getShift()
    timeNow = datetime.datetime.now() + datetime.timedelta(hours = 5.5)   
    print(old_shift)
    if shift == old_shift:
        old_shift = shift
        print(old_shift)
        data= pd.read_sql_query('''select machineId,shift from person_detection where dateTime >= date_sub(CONVERT_TZ(NOW(),'+00:00','+05:30'),INTERVAL 1 DAY);''',db_conn[1])
        df = pd.DataFrame(data)
        machine_id =  df['machineId'] == 'FG2/U+2/Top0'
        for i in machine_id:
            print("machine_id",i)
            if timeNow.hour>19 or timeNow.hour<20 and i == True:
                flag = 1
            else:
                flag = 0
            
            return flag
    
results = check_offset_lunch()
print(results)