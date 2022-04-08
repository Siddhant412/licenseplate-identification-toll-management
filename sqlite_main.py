import random
import string
import sqlite3

def id_generator(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

veh_type = ['Car','LCV','Trailer','Bus','Truck','3-wheeler','MAV']

conn = sqlite3.connect('lp_tollpayment.db')

# conn.execute("""
# CREATE TABLE Lp_owner
# (
#     lp_number VARCHAR PRIMARY KEY,
#     owner_name VARCHAR(40),
#     vehicle_type VARCHAR,
#     account_number VARCHAR(40),
#     balance INT 
# )
# """)
# print("Database created successfully")

# for i in range(100,9999):
#     lpno = 'MH05'+ str(i) 
#     name = id_generator()
#     type = veh_type[random.randint(0,6)]
#     accno = str(10000+i)
#     balance = str(2000 + i)
#     query = "INSERT INTO Lp_owner (lp_number,owner_name,vehicle_type,account_number,balance) values('"+lpno+"','"+name+"','"+type+"','"+accno+"','"+balance+"')"
#     print(query)
#     conn.execute(query)
# print("record added successfully")

# conn.execute("INSERT INTO Lp_owner values('M1H05DS1058','Abhijeet','Car','12345','2000');")
# conn.commit()

# delete all records 
# conn.execute("DELETE from Lp_owner;")
# conn.commit()


# print all entries 
# cursor = conn.execute("SELECT * from Lp_owner")
# for row in cursor:
#     print(row[0],row[1],row[2],row[3])


cursor = conn.execute("SELECT * from Lp_owner where lp_number='M1H05DS1058'")
for row in cursor:
    print(row[0],row[1],row[2],row[3],row[4])


conn.close()