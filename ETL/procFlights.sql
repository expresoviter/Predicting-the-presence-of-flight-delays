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