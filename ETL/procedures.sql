CREATE OR REPLACE PROCEDURE stagezone.processAirlines()
language plpgsql
as $$

declare 
	rec record;
	lineCursor cursor for select * from stagezone.airlines;
	
begin
	open lineCursor;
	loop
		fetch lineCursor into rec;
		exit when not found;
		if rec.iatacode is not null and rec.airlineName is not null 
		and rec.iatacode not in (SELECT iatacode FROM warehouse.dim_airlines) then
			insert into warehouse.dim_airlines(iatacode,airlineName) values
			(rec.iatacode,rec.airlineName);
		end if;
	end loop;
	close lineCursor;
end;$$;

CREATE OR REPLACE PROCEDURE stagezone.processAirports()
language plpgsql
as $$

declare 
	rec record;
	lineCursor cursor for select * from stagezone.airports;
	forID integer;
	isCorrectVal bool;
	
begin
	open lineCursor;
	loop
		fetch lineCursor into rec;
		exit when not found;
		
		
		if rec.country is not null 
		and rec.country not in (SELECT countryname FROM warehouse.dim_countries) then
			insert into warehouse.dim_countries(countryname) values
			(rec.country);
		end if;
		
		if rec.state not in (SELECT statecode FROM warehouse.dim_states) then
			forid:=(SELECT id FROM warehouse.dim_countries
				   WHERE warehouse.dim_countries.countryname=rec.country);
			insert into warehouse.dim_states(country_id,statecode) values
			(forid, rec.state);
		end if;
		
		
		
		if rec.city is not null
		and rec.city not in (SELECT cityname FROM warehouse.dim_cities WHERE state_id=(
		select id from warehouse.dim_states where statecode=rec.state)) then
			forid:=(SELECT warehouse.dim_states.id FROM warehouse.dim_states, warehouse.dim_countries
				   WHERE warehouse.dim_states.statecode=rec.state
				   and warehouse.dim_states.country_id=(
				   SELECT warehouse.dim_countries.id
					   WHERE warehouse.dim_countries.countryname=rec.country));
			insert into warehouse.dim_cities(state_id,cityname) values
			(forid, rec.city);
		end if;
		
		if rec.iatacode is not null 
		and (rec.iatacode not in (SELECT iatacode FROM warehouse.dim_airports) or
			 rec.airportName not in (SELECT airportName FROM warehouse.dim_airports WHERE iatacode=rec.iatacode))
			 then
			forid:=(SELECT warehouse.dim_cities.id FROM warehouse.dim_cities, warehouse.dim_states
				   WHERE warehouse.dim_cities.cityname=rec.city
				   and warehouse.dim_cities.state_id=(
				   SELECT warehouse.dim_states.id
					   WHERE warehouse.dim_states.statecode=rec.state));
			if rec.iatacode in (SELECT iatacode FROM warehouse.dim_airports)
				then
					update warehouse.dim_airports
					set isCorrect=False
					where iatacode=rec.iatacode and isCorrect=True;
			end if;
			insert into warehouse.dim_airports(city_id,iatacode,airportName,
											  latitude,longitude,isCorrect) values
			(forid,rec.iatacode,rec.airportName,rec.latitude,rec.longitude,True);
		end if;
	end loop;
	close lineCursor;
end;$$

CREATE OR REPLACE PROCEDURE stagezone.processFlights()
language plpgsql
as $$

declare 
	rec record;
	lineCursor cursor for select * from stagezone.flights;
	forID integer;
	
begin
	open lineCursor;
	loop
		fetch lineCursor into rec;
		exit when not found;		
		
		if rec.day not in (SELECT day FROM (
			SELECT * FROM (SELECT * FROM warehouse.dim_dates WHERE year=rec.year) as y WHERE month=rec.month) as mo) then
			insert into warehouse.dim_dates(year,month,day,weekday) values
			(rec.year,rec.month,rec.day,rec.weekday);
		end if;
		
		if rec.tailnumber is not null and
		rec.tailnumber not in (select tailnumber from warehouse.dim_aircraft) then
			insert into warehouse.dim_aircraft(tailnumber) values
			(rec.tailnumber);
		end if;
		
		if rec.cancelreason is not null and 
		rec.cancelreason not in (select cancelreason from warehouse.dim_cancelreason) then
			insert into warehouse.dim_cancelreason(cancelreason) values
			(rec.cancelreason);
		end if;
		
		
	end loop;
	close lineCursor;
end;$$

CREATE OR REPLACE PROCEDURE stagezone.processfacts()
language plpgsql
as $$

declare 
	rec record;
	lineCursor cursor for select * from stagezone.flights;
	dateID integer;
	airlineId integer;
	aircraftId integer;
	origAir integer;
	destAir integer;
	reasonID integer;
	
begin
	open lineCursor;
	loop
		fetch lineCursor into rec;
		exit when not found;
		
		
		SELECT id from warehouse.dim_dates where year=rec.year and month=rec.month and day=rec.day
		into dateid;
		
		SELECT id from warehouse.dim_airlines where iatacode=rec.airline
		into airlineid;
		
		SELECT id from warehouse.dim_aircraft where tailnumber=rec.tailnumber
		into aircraftid;
		
		SELECT id from warehouse.dim_airports where iatacode=rec.originairport and isCorrect=True
		into origair;
		
		SELECT id from warehouse.dim_airports where iatacode=rec.destairport
		into destair;
		
		SELECT id from warehouse.dim_cancelreason where cancelreason=rec.cancelreason
		into reasonId;
		
		insert into warehouse.fact_flights(dimDatesId,dimAirlinesId,dimAircraftId,
											   originAirportId,destAirportId,dimCancelReasonId,
											   scheduledDep,departureTime,departureDelay,taxiOut,
											   wheelsOff,scheduledTime,elapsedTime,airTime,distance,
											   wheelsOn,taxiIn,scheduledArrival,arrivalTime,
											   arrivalDelay,diverted,cancelled,systemDelay,
											   securityDelay,airlineDelay,lateDelay,weatherDelay)
			values (dateID,airlineId,aircraftId,origAir,destAir,reasonID,
				   rec.scheduledDep,rec.departureTime,rec.departureDelay,
					rec.taxiOut,rec.wheelsOff,rec.scheduledTime,rec.elapsedTime,
					rec.airTime,rec.distance,rec.wheelsOn,rec.taxiIn,rec.scheduledArrival,
					rec.arrivalTime,rec.arrivalDelay,rec.diverted,rec.cancelled,rec.systemDelay,
					rec.securityDelay,rec.airlineDelay,rec.lateDelay,rec.weatherDelay);
	end loop;
	close lineCursor;
end;$$