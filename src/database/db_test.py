import ConnectDb
from ConnectDb import postgreSql
import psycopg2
from ConnectDb import DatabaseHandler
import datetime
import pandas as pd
# cols = ['title','genre','director','release_year']
# colsvalues = ["strange", "psychological thriller", "Todd Phillips", 2022]

# db = push_into_db()
# db.run(cols,colsvalues)

# db = sql()
# #db_conn = db.connect_with_sql("admin", "Dev123456", "hzl.cgwyhisjxon9.ap-south-1.rds.amazonaws.com", "3306", "hzl")
# db_conn = db.connect_with_sql("root", "root123", "localhost", "3306", "ivision")
# print(type(db_conn))
# #print(type(db_conn[1]))
# query = "show tables;"
# print(db.fetch_details(query, db_conn[1]))
# now = datetime.datetime.now()
# shiftstart = '6:00:00'
# shiftend = '14:00:00'
# scheduleHours = 8
# shift = 'A'
# machine = "FG2/U+2/Top3"
# # query1 = "INSERT INTO `person_detection`(index, dateTime, shift,machineId,shiftstart,shiftEnd,scheduleHours) VALUES (0,now,shift,machine,shiftstart,shiftend,scheduleHours);"
                                
# # query1 = "INSERT INTO `person_detection` VALUES (0,'2022-09-15 22:03:08.004434', 'A',"FG2/U+2/Top3",'6:00:00','14:00:00',7.5);"
# # db.run_query(query1,db_conn[1])


# data= pd.read_sql_query('''select machineId,shift from person_detection where dateTime >= date_sub(CONVERT_TZ(NOW(),'+00:00','+05:30'),INTERVAL 1 DAY);''',db_conn[1])

# df = pd.DataFrame(data)
# machine = df['machineId']== 'FG2/U+2/Top3'
# for i in machine:
#     if i == True:
#         print("done")
# # machine= df[df['machineId'] == 'FG2/U+2/Top3']['machineId'].item()
# # df_filt = df[df['machineId'] == 'FG2/U+2/Top3' & df['shift'] == 'A']
# # result = df_filt.get_value(df_filt.index[0],'VALUE')
# # print(result)
# print("DataFrame Fetched with shape: ", df.shape)
#db.insertRow(tablename = "test", data ={'roll_id':'251452', 'name': 'Python', 'age': 28})
#
## var = ConnectDb.connect_with_sql("admin", "Dev123456", "hzl.cgwyhisjxon9.ap-south-1.rds.amazonaws.com", "3306", "hzl")
## print(var)
## query = "select * from stacking;"
## print(ConnectDb.fetch_details(query, var[1]))
#
#db_postgresql = postgreSql()
#conn = db_postgresql.connect_postgresql()
#print(type(conn))
#print(type(conn[1]))
#query = ("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (27, 'vik', 32, 'California', 20000.00 )");
## print(query)
#query1 =("SELECT id, name, address, salary  from COMPANY")
## query = obj.insertRow("COMPANY", {'ID':25, 'NAME': 'Python', 'AGE': 28,'ADDRESS': 'California', 'SALARY':46543546213},conn[1])
## print(query)
#print(db_postgresql.run_query(query,conn[1]))
#print(db_postgresql.fetch_details(query1,conn[1]))
# query2 = ("INSERT INTO COMPANY (ID, NAME, AGE, ADDRESS, SALARY) VALUES (23, 'Python', 28, 'California', 46543546213)");
# print(query2)
# print(obj.run_query(query2,conn[1]))
# # sqlstatement = 'SELECT * FROM [dbo].[ObservationData]  as obs where obs.TankSystemId in '+ str(deliveryDict) +' and obs.TimeStamp in '+ str(timestamp_set)


# def insertRow(self,*args,**kwargs):

#       keys = []
#       values = []
#       for k,v in d.items():
#           keys.append(str(k))
#           values.append(str(v))
#       sql = "insert into {1} {2} values{3}".format(str(kwargs.keys()),str(kwargs.values())) 

#       try:
#             self.cursor.execute(sql)
#             self.dbs.commit()
#       except Exception as e:
#             print("Error during insert:::",e)


db = DatabaseHandler()

query = "select * from cam_config;"

data = db.fetch_data(query)
print(data)