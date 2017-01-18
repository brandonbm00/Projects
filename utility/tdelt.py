from dateutil.relativedelta import relativedelta
from datetime import datetime
from math import floor

def add_years_to_age(date, age):
    first = datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
    last = datetime.now()

    return age + floor(last.year - first.year - ((last.month, last.day) < (first.month, first.day)) )


if __name__ == '__main__':
    testdate = '19911120'
    print(add_years_to_age(testdate, 0))
