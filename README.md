# NYC Taxi Analysis (August 2024)

looking at nyc taxi data for august 2024 to check traffic patterns and make predictions

## libraries used

* pandas for data handling
* numpy for calculations
* sklearn for random forest model
* matplotlib and seaborn for plots
* networkx for route mapping
* pyarrow for reading parquet files

## model info

used random forest because:

* handles both numbers and categories well (like locations and times)
* deals with messy taxi data better than linear models
* can tell us which stuff matters most
* works better for nyc traffic since its not simple linear patterns

model stats explained:

* R2: 0.83 (means model explains 83% of whats happening with trip times)
* MSE: 0.0091 (average squared error in hours, pretty small)
* MAE: 0.0576 (average off by about 3.5 minutes)

what affects predictions:

* trip distance (87% importance): obviously longer trips take longer
* time of day (5%): rush hour vs off peak
* pickup/dropoff spots (3% each): some areas slower than others
* other stuff like weekday/weekend matter less

## what i found

busy spots (august):

* jamaica & upper east side most trips
* midtown east packed
* crown heights north slowest (8.3 mph)
* upper east side lots of internal rides

traffic stuff:

* worst at 5-6pm
* best at 5am
* weekends way different
* most trips under 5mi
* mondays cost more

## running it

put data in folder and run script, plots go to results

thats it lmk if you need help
