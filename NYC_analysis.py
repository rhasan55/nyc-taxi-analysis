import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import networkx as nx
import seaborn as sns
import os

class NYCAnalyzer:
    def __init__(self):
        self.data = None
        self.rf_model=None
        self.scaler = RobustScaler()
        self.graph = nx.DiGraph()
        self.zones={
            61:"Crown Heights North",186:"Astoria",161:"Jamaica",
            100:"Clinton East",234:"Williamsburg South",163:"Midtown Center",
            68:"East Chelsea",113:"Greenwich Village South",164:"Midtown South",
            162:"Midtown East",237:"Upper East Side South",236:"Upper East Side North",
            151:"Long Island City",142:"Lincoln Square East",238:"Upper West Side South",
            239:"Upper West Side North",48:"Clinton West",50:"Pavement Edge"
        }
        os.makedirs('results/figs',exist_ok=True)
    
    def load_data(self):
        self.data=pd.read_parquet('yellow_tripdata_2024-08.parquet')
        self._clean_data()
        self._make_graph()
        self.data['pickup_zone']=self.data['PULocationID'].map(self.zones)
        self.data['dropoff_zone']=self.data['DOLocationID'].map(self.zones)
    
    def _clean_data(self):
        self.data['pickup_time']=pd.to_datetime(self.data['tpep_pickup_datetime'])
        self.data['dropoff_time']=pd.to_datetime(self.data['tpep_dropoff_datetime'])
        
        self.data['hour']=self.data['pickup_time'].dt.hour
        self.data['day']=self.data['pickup_time'].dt.dayofweek
        self.data['weekend']= self.data['day'].isin([5,6]).astype(int)
        self.data['rush']= (((self.data['hour']>=7)&(self.data['hour']<=10))|
            ((self.data['hour']>=16)&(self.data['hour']<=19))).astype(int)
        
        trip_hrs=(self.data['dropoff_time']-self.data['pickup_time']).dt.total_seconds()/3600
        self.data['duration']=trip_hrs
        
        good_trips=(self.data['trip_distance']>0)&(self.data['trip_distance']<50)&(self.data['duration']>0)&(self.data['duration']<3)&(self.data['total_amount']>0)&(self.data['passenger_count']>0)&(self.data['passenger_count']<=6)
        self.data=self.data[good_trips]
        
        self.data['speed']=self.data['trip_distance']/self.data['duration']
        max_speed=self.data['speed'].quantile(0.99)
        self.data=self.data[self.data['speed']<max_speed]
        
        self.data['price_mile']=self.data['total_amount']/self.data['trip_distance']
        self.data=self.data.replace([np.inf,-np.inf],np.nan).dropna()

    def _make_graph(self):
        pairs = self.data.groupby(['PULocationID','DOLocationID']).agg({
            'duration':['mean','count'],
            'speed':'mean',
            'trip_distance':'mean'
        }).reset_index()
        
        pairs.columns = ['pickup','dropoff','time','trips','speed','dist']
        pairs=pairs[pairs['trips']>50]
        
        for _,r in pairs.iterrows():
            self.graph.add_edge(
                int(r['pickup']),int(r['dropoff']),
                weight=float(r['time']),speed=float(r['speed']),
                distance=float(r['dist']),volume=int(r['trips'])
            )

    def make_plots(self):
        fig,ax=plt.subplots(2,2,figsize=(15,12))
        
        trips=self.data.groupby('hour').size()
        ax[0,0].bar(trips.index,trips.values)
        ax[0,0].set_title('Trips by Hour')
        ax[0,0].set_xlabel('Hour')
        ax[0,0].set_ylabel('Trip Count')
        
        spd=self.data.groupby(['hour','weekend'])['speed'].mean().unstack()
        ax[0,1].plot(spd.index,spd[0],'r-',label='Weekday')
        ax[0,1].plot(spd.index,spd[1],'b--',label='Weekend')
        ax[0,1].set_title('Speed by Hour')
        ax[0,1].set_xlabel('Hour')
        ax[0,1].set_ylabel('Avg Speed (mph)')
        ax[0,1].legend()
        
        ax[1,0].hist(self.data['trip_distance'],bins=50)
        ax[1,0].set_title('Trip Distances')
        ax[1,0].set_xlabel('Miles')
        ax[1,0].set_ylabel('Count')
        
        fare=self.data.groupby('day')['total_amount'].mean()
        ax[1,1].bar(fare.index,fare.values)
        ax[1,1].set_title('Daily Fares')
        ax[1,1].set_xlabel('Day (0=Mon)')
        ax[1,1].set_ylabel('Avg Fare ($)')
        plt.tight_layout()
        plt.savefig('results/figs/traffic.png')
        plt.close()

    def train(self):
        features=['hour','day','trip_distance','passenger_count','weekend','rush','PULocationID','DOLocationID']
        X=self.data[features]
        y=self.data['duration']
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        X_train_scaled=self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)
        
        self.rf_model=RandomForestRegressor(n_estimators=200,max_depth=15,min_samples_split=10,min_samples_leaf=5,random_state=42)
        self.rf_model.fit(X_train_scaled,y_train)
        preds=self.rf_model.predict(X_test_scaled)
        
        mse=mean_squared_error(y_test,preds)
        mae=mean_absolute_error(y_test,preds)
        r2=r2_score(y_test,preds)
        
        plt.figure(figsize=(10,6))
        plt.scatter(y_test,preds,alpha=0.5,s=1)
        plt.plot([0,y_test.max()],[0,y_test.max()],'r--')
        plt.xlabel('Actual Time')
        plt.ylabel('Predicted Time')
        plt.title('Predictions vs Actual')
        plt.savefig('results/figs/preds.png')
        plt.close()
        
        feature_imp=pd.DataFrame({
            'feature':features,
            'importance':self.rf_model.feature_importances_
        }).sort_values('importance',ascending=False)
        
        return{'mse':mse,'mae':mae,'r2':r2,'feature_importance':feature_imp}

    def check_zones(self):
        print("\nPopular Pickup Areas:")
        pickups = self.data['PULocationID'].value_counts().head(10)
        for z,cnt in pickups.items():
            name=self.zones.get(z,"Unknown")
            print(f"Zone {z} ({name}): {cnt} pickups")
            
        print("\nPopular Dropoff Areas:")
        drops=self.data['DOLocationID'].value_counts().head(10)
        for z,cnt in drops.items():
            name=self.zones.get(z,"Unknown")
            print(f"Zone {z} ({name}): {cnt} dropoffs")
    
    def check_routes(self):
        print("\nBusiest Routes:")
        popular=self.data.groupby(['PULocationID','DOLocationID']).size().sort_values(ascending=False).head(10)
        for (p,d),cnt in popular.items():
            p_name=self.zones.get(p,"Unknown")
            d_name=self.zones.get(d,"Unknown")
            print(f"{p_name} → {d_name}: {cnt} trips")

    def plot_busy_zones(self):
        zones=self.data.groupby('PULocationID').agg({
            'speed':'mean',
            'duration':'count',
            'rush':['mean',lambda x:(x*x.size).sum()/x.size]
        }).round(3)
        
        zones.columns=['speed','trips','rush_pct','rush_vol']
        busy=zones[zones['trips']>1000].sort_values('speed')
        
        plt.figure(figsize=(12,6))
        plt.scatter(busy['rush_pct'],busy['speed'],alpha=0.5,s=busy['trips']/100)
        plt.xlabel('Rush Hour %')
        plt.ylabel('Speed (mph)')
        plt.title('Zone Analysis')
        plt.savefig('results/figs/zones.png')
        plt.close()
        
        return busy.head(10)

    def find_route(self,start,end,time):
        start_name=self.zones.get(start,"Unknown")
        end_name=self.zones.get(end,"Unknown")
        
        print(f"\nRoutes from {start_name} (Zone {start}) to {end_name} (Zone {end})")
        
        if not (self.graph.has_node(start) and self.graph.has_node(end)):
            print(f"\nAnalyzing available connections:")
            print(f"Pickup zone has {len(list(self.graph.edges(start)))} outgoing routes")
            print(f"Dropoff zone has {len(list(self.graph.in_edges(end)))} incoming routes")
            return None
            
        try:
            all_paths = []
            for path in nx.shortest_simple_paths(self.graph,int(start),int(end),weight='weight'):
                if len(all_paths) >= 3:
                    break
                all_paths.append(path)
                
            routes=[]
            for p in all_paths:
                total_time=0
                total_dist=0
                segs=[]
                
                for i in range(len(p)-1):
                    edge=self.graph[p[i]][p[i+1]]
                    t=edge['weight']
                    if time in [7,8,9,16,17,18,19]: 
                        t*=1.3
                    
                    from_name=self.zones.get(p[i],"Unknown")
                    to_name=self.zones.get(p[i+1],"Unknown")
                    
                    total_time+=t
                    total_dist+=edge['distance']
                    segs.append({
                        'from_id':p[i],'to_id':p[i+1],
                        'from':from_name,'to':to_name,
                        'time':t,'dist':edge['distance'],
                        'speed':edge['speed']
                    })
                    
                routes.append({
                    'path':p,'segs':segs,
                    'time':total_time,'dist':total_dist,
                    'speed':total_dist/total_time
                })
            
            return sorted(routes,key=lambda x:x['time'])
            
        except nx.NetworkXNoPath:
            print("\nNo direct path found. Analyzing alternatives:")
            
            midpoints = set(self.graph.successors(start)) & set(self.graph.predecessors(end))
            if midpoints:
                print("\nPossible intermediate points:")
                for z in list(midpoints)[:3]:
                    name=self.zones.get(z,"Unknown")
                    print(f"- {name} (Zone {z})")
                    
            return None

if __name__=="__main__":
    nyc=NYCAnalyzer()
    nyc.load_data()
    
    print("Making plots...")
    nyc.make_plots()
    
    print("Training model...")
    stats=nyc.train()
    print(f"\nModel Performance:")
    print(f"MSE: {stats['mse']:.4f}")
    print(f"MAE: {stats['mae']:.4f}")
    print(f"R2: {stats['r2']:.4f}")
    print("\nTop 5 Important Features:")
    print(stats['feature_importance'].head())

    print("\nChecking zones...")
    nyc.check_zones()
    nyc.check_routes()
    
    busy=nyc.plot_busy_zones()
    print("\nBusiest areas:")
    print(busy)
    
    print("\nTesting routes...")
    routes=nyc.find_route(162,237,9)
    if routes:
        print("\nBest routes found:")
        for i,r in enumerate(routes[:3],1):
            print(f"\nRoute {i}:")
            path_names=[f"{nyc.zones.get(z,'Unknown')} ({z})" for z in r['path']]
            print(" → ".join(path_names))
            print(f"Est. time: {r['time']:.2f} hours")
            print(f"Distance: {r['dist']:.2f} miles")
            print(f"Avg speed: {r['speed']:.2f} mph")
            
            print("\nSegment details:")
            for s in r['segs']:
                print(f"- {s['from']} → {s['to']}")
                print(f"  {s['dist']:.2f} miles, {s['speed']:.2f} mph")