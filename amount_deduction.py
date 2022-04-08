import sqlite3
import json

costs = {
    'Car':50,
    'LCV':60,
    'Trailer':100,
    'Bus':90,
    'Truck':80,
    '3-wheeler':30,
    'MAV':100
}

conn = sqlite3.connect('lp_tollpayment.db')

with open('lp_data.json', 'r') as openfile:
    json_object = json.load(openfile)

for lpnos in json_object.values():
    vehicle_type = conn.execute("select vehicle_type from Lp_owner where lp_number = '"+lpnos+"'")
    type = [i for i in vehicle_type]
    print(type[0][0])
    print(lpnos)
    query = "UPDATE Lp_owner set balance = balance - "+str(costs[type[0][0]])+" where lp_number = '"+lpnos+"'"
    conn.execute(query)
    conn.commit()
    print("Amount deducted")