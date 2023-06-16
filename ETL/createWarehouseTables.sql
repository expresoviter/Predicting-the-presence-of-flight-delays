create schema warehouse;

create table warehouse.dim_Countries(
id serial primary key,
countryName varchar(50) unique not null);

create table warehouse.dim_States(
id serial primary key,
country_id integer,
stateCode varchar(2) unique,
foreign key (country_id) references warehouse.dim_Countries(id));

create table warehouse.dim_Cities(
id serial primary key,
state_id integer,
cityName varchar(50) not null,
foreign key (state_id) references warehouse.dim_States(id));

create table warehouse.dim_Airports(
id serial primary key,
city_id integer,
IATAcode varchar(3) unique not null,
airportName varchar(100) unique not null,
latitude real,
longitude real,
isCorrect bool,
foreign key (city_id) references warehouse.dim_Cities(id));

create table warehouse.dim_Airlines(
id serial primary key,
IATAcode varchar(2) unique not null,
airlineName varchar(100) unique not null);

create table warehouse.dim_Dates(
id serial primary key,
year integer,
month integer,
day integer,
weekday integer);

create table warehouse.dim_Aircraft(
id serial primary key,
tailnumber varchar(10) unique not null);

create table warehouse.dim_CancelReason(
id serial primary key,
cancelReason varchar(1) unique);

create table warehouse.fact_Flights(
id serial primary key,
dimDatesId integer,
dimAirlinesId integer,
dimAircraftId integer,
originAirportId integer,
destAirportId integer,
dimCancelReasonId integer,

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
  systemDelay integer,
  securityDelay integer,
  airlineDelay integer,
  lateDelay integer,
  weatherDelay integer,

	foreign key (dimDatesId) references warehouse.dim_dates(id),
	foreign key (dimAirlinesId) references warehouse.dim_airlines(id),
	foreign key (dimAircraftId) references warehouse.dim_aircraft(id),
	foreign key (originAirportId) references warehouse.dim_airports(id),
	foreign key (destAirportId) references warehouse.dim_airports(id),
	foreign key (dimCancelReasonId) references warehouse.dim_cancelreason(id)
);