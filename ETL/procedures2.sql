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