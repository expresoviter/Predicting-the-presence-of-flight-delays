create schema stageZone;

create table stagezone.airlines (
id serial primary key,
IATAcode varchar(2) unique not null,
airlineName varchar(100) not null);

create table stagezone.airports(
id serial primary key,
IATAcode varchar(3) unique not null,
airportName varchar(100) not null,
city varchar(50) not null,
state varchar(2) not null,
country varchar(50) not null,
latitude real,
longitude real);

create table stagezone.flights(
id serial primary key,
year integer not null,
month integer not null,
day integer not null,
weekday integer,
airline varchar(2) not null,
flightNumber integer not null,
tailNumber varchar(10),
originAirport varchar(3) not null,
destAirport varchar(3) not null,

scheduledDep integer,
departureTime integer,
  departureDelay integer,
  taxiOut integer,
  wheelsOff integer,
  scheduledTime integer,
  elapsedTime integer,
  airTime integer,
  distance integer not null,
  wheelsOn integer,
  taxiIn integer,
  scheduledArrival integer,
  arrivalTime integer,
  arrivalDelay integer,
  diverted boolean not null,
  cancelled boolean not null,
  cancelReason varchar(1),
  systemDelay integer,
  securityDelay integer,
  airlineDelay integer,
  lateDelay integer,
  weatherDelay integer);
  
copy stagezone.airlines(iatacode,airlinename)
from 'C:\csvFiles\airlines.csv'
delimiter ','
csv header;

copy stagezone.airports(iatacode,airportname,city,state,country,latitude,longitude)
from 'C:\csvFiles\airports.csv'
delimiter ','
csv header;

copy stagezone.flights(year,month,day,weekday,airline,flightnumber,tailnumber,originairport,
            destairport,scheduleddep,departuretime,departuredelay,
            taxiout,wheelsoff,scheduledtime,elapsedtime,airtime,distance,
            wheelson,taxiin,scheduledarrival,arrivaltime,arrivaldelay,
            diverted,cancelled,cancelreason,systemdelay,securitydelay,
            airlinedelay,latedelay,weatherdelay)
from 'C:\csvFiles\flights1.csv'
delimiter ','
csv header;