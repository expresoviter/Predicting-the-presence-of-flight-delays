from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Airline(Base):
    __tablename__ = 'dim_airlines'
    id = Column(Integer, primary_key=True)
    iatacode = Column(String)
    airlinename = Column(String)
    __table_args__ = {'schema': 'warehouse'}

    def __init__(self, iatacode, airlinename):
        self.iatacode = iatacode
        self.airlinename = airlinename

    def __repr__(self):
        return self.airlinename


class Date(Base):
    __tablename__ = 'dim_dates'
    id = Column(Integer, primary_key=True)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    weekday = Column(Integer)
    __table_args__ = {'schema': 'warehouse'}

    def __init__(self, year, month, day, weekday):
        self.year = year
        self.month = month
        self.day = day
        self.weekday = weekday

    def __repr__(self):
        return f'{self.year}-{self.month}-{self.day}'


class Flight(Base):
    __tablename__ = 'fact_flights'
    id = Column(Integer, primary_key=True)
    dimdatesid = Column(Integer, ForeignKey('warehouse.dim_dates.id'))
    dimairlinesid = Column(Integer, ForeignKey('warehouse.dim_airlines.id'))
    dimaircraftid = Column(Integer, ForeignKey('warehouse.dim_aircraft.id'))
    originairportid = Column(Integer, ForeignKey('warehouse.dim_airports.id'))
    destairportid = Column(Integer, ForeignKey('warehouse.dim_airports.id'))
    dimcancelreasonid = Column(Integer, ForeignKey('warehouse.dim_cancelreason.id'))
    scheduleddep = Column(Integer)
    departuretime = Column(Integer)
    departuredelay = Column(Integer)
    taxiout = Column(Integer)
    wheelsoff = Column(Integer)
    scheduledtime = Column(Integer)
    elapsedtime = Column(Integer)
    airtime = Column(Integer)
    distance = Column(Integer)
    wheelson = Column(Integer)
    taxiin = Column(Integer)
    scheduledarrival = Column(Integer)
    arrivaltime = Column(Integer)
    arrivaldelay = Column(Integer)
    diverted = Column(Boolean)
    cancelled = Column(Boolean)
    systemdelay = Column(Integer)
    securitydelay = Column(Integer)
    airlinedelay = Column(Integer)
    latedelay = Column(Integer)
    weatherdelay = Column(Integer)
    __table_args__ = {'schema': 'warehouse'}

    def __init__(self, dimDatesId, dimAirlinesId, dimAircraftId,
                 originAirportId, destAirportId, dimCancelReasonId,
                 scheduledDep, departureTime, departureDelay, taxiOut,
                 wheelsOff, scheduledTime, elapsedTime, airTime, distance,
                 wheelsOn, taxiIn, scheduledArrival, arrivalTime, arrivalDelay,
                 diverted, cancelled, systemDelay, securityDelay, airlineDelay, lateDelay, weatherDelay):
        self.dimDatesId = dimDatesId
        self.dimAirlinesId = dimAirlinesId
        self.dimAircraftId = dimAircraftId
        self.originAirportId = originAirportId
        self.destAirportId = destAirportId
        self.dimCancelReasonId = dimCancelReasonId
        self.scheduledDep = scheduledDep
        self.departureTime = departureTime
        self.departureDelay = departureDelay
        self.taxiOut = taxiOut
        self.wheelsOff = wheelsOff
        self.scheduledTime = scheduledTime
        self.elapsedTime = elapsedTime
        self.airTime = airTime
        self.distance = distance
        self.wheelsOn = wheelsOn
        self.taxiIn = taxiIn
        self.scheduledArrival = scheduledArrival
        self.arrivalTime = arrivalTime
        self.arrivalDelay = arrivalDelay
        self.diverted = diverted
        self.cancelled = cancelled
        self.systemDelay = systemDelay
        self.securityDelay = securityDelay
        self.airlineDelay = airlineDelay
        self.lateDelay = lateDelay
        self.weatherDelay = weatherDelay
