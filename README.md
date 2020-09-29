# retrosheetsetl

In order to properly load into the SQL database you will need to change your information in the models/db_utils_sample.py document 
and rename the file db_utils.py
You will need to change username, password, ipaddr, and register

To begin the environment so that you have the correct packages run the command:

./retrosheet-env/Scripts/Activate.ps1

Once that is active you can run:

./retrosheet-etl.py -y YEAR

-y flag - any 4-digit year for which retrosheet has MLB data (since 1950 for best results)

This program will get play by play statistics for every plate appearance in the given year and load them into the SQL
database as specified by models/sqla_utils.py

Once it has completed that it will use the database to calculate player, team, and league statistics for the given year
and load these statistics into the database.

The program takes quite a while to run. Expect 15-30 minutes for one year.

Separately you can run:
./playoff-etl.py -y YEAR

-y flag - any 4-digit year for which retrosheet has MLB data (since 1950 for best results)

This proram will do the same as the above but only for the given year's playoff games