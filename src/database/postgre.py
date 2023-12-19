import psycopg2

# conn = psycopg2.connect(database = "testdb", user = "postgres", password = "pass123", host = "127.0.0.1", port = "5432")
# print(conn)
# print("Opened database successfully")

# cur = conn.cursor()
# cur.execute('''CREATE TABLE COMPANY
#       (ID INT PRIMARY KEY     NOT NULL,
#       NAME           TEXT    NOT NULL,
#       AGE            INT     NOT NULL,
#       ADDRESS        CHAR(50),
#       SALARY         REAL);''')
# print("Table created successfully")

# conn.commit()
# conn.close()

# def connect_with_postgreSql(database,user, password, host, port):
#       '''
#         ### Args:
#             sql_username : username of sql
#             sql_password : password of sql
#             sql_ip       : ip of sql
#             sql_port     : portn number(3306)
#             sql_database : databases
        
#         ### Returns:
#             -------
#             engine : True or False
#             ------
#       '''        
#       try:
#             connect_query = psycopg2.connect(database,user,password,host,port)
#             cursor = connect_query.cursor()
#             success_log(200, "Connection Sucessfully Created", "connect_with_postgresql", path)
#             return (True, cursor)
#       except:
#             return (False, "Failed to connect")
#             error_log(400, "Failed to connect", "connect_with_postgresql", path)



# var = connect_with_postgreSql(database = "testdb", user = "postgres", password = "pass123", host = "127.0.0.1", port = "5432")
# print(var)
#!/usr/bin/python
# import psycopg2
# from config import config

# def connect_postgresql():
#     """ Connect to the PostgreSQL database server """
#     conn = None
#     try:
#         # read connection parameters
#         params = config()

#         # connect to the PostgreSQL server
#         print('Connecting to the PostgreSQL database...')
#         conn = psycopg2.connect(**params)
            
#         # create a cursor
#         cur = conn.cursor()
#         print("connection done")
#         return cur
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)



# var = connect_postgresql()
# print(var)
import psycopg2

conn = psycopg2.connect(database = "testdb", user = "postgres", password = "pass123", host = "127.0.0.1", port = "5432")
print("Opened database successfully")

cur = conn.cursor()

cur.execute("SELECT id, name, address, salary  from COMPANY")
rows = cur.fetchall()
for row in rows:
   print("ID = ", row[0])
   print("NAME = ", row[1])
   print("ADDRESS = ", row[2])
   print("SALARY = ", row[3]), "\n"

print("Operation done successfully")
conn.close()

# import psycopg2

# conn = psycopg2.connect(database = "testdb", user = "postgres", password = "pass123", host = "127.0.0.1", port = "5432")
# print("Opened database successfully")

# cur = conn.cursor()

# cur.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
#       VALUES (9, 'Paul', 32, 'California', 20000.00 )");

# cur.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
#       VALUES (10, 'Allen', 25, 'Texas', 15000.00 )");

# cur.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
#       VALUES (11, 'Teddy', 23, 'Norway', 20000.00 )");

# cur.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
#       VALUES (12, 'Mark', 25, 'Rich-Mond ', 65000.00 )");

# conn.commit()
# print("Records created successfully")
# # #conn.close()