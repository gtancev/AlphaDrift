# AlphaDrift

## Example

### Load libriaries.

    import pandas as pd
    import numpy as np
    from AlphaDrift import Sensor, AlphaDrift
    from sklearn.covariance import EllipticEnvelope
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    
### Load data.

    data = pd.read_csv("../folder/file.csv", index_col=0)

### Specify sensors.

    sensors = [Sensor(s=[[0.42, 0.65], [0.0, 0.0], [0.0, 0.0]],
                      z=[-250.0, 30.0],
                      d_s=[[0.9, 1.0], [1.0, 1.0], [1.0, 1.0]],
                      d_z=[-50.0, 50.0],
                      c=[]), 
               Sensor(s=[[0.0, 0.0], [-0.65, -0.20], [0.0, 0.0]],
                      z=[-80.0, 80.0],
                      d_s=[[1.0, 1.0], [0.6, 0.8], [1.0, 1.0]],
                      d_z=[-12.0, 12.0],
                      c=[]), 
               Sensor(s=[[0.0, 0.0], [-0.75, -0.23], [-0.75, -0.23]],
                      z=[-80.0, 80.0],
                      d_s=[[1.0, 1.0], [0.6, 0.8], [0.6, 0.8]],
                      d_z=[-10.0, 10.0],
                      c=[1, 2])]

### Define model.
    
    model = AlphaDrift(sensors)

### Load detectors.

    contamination = 0.05
    RC = EllipticEnvelope(contamination=contamination, random_state=0)
    LOF = LocalOutlierFactor(n_neighbors=200, p=2, contamination=contamination, novelty=True, n_jobs=-1)
    OCSVM = OneClassSVM(kernel="rbf", gamma="auto", nu=contamination)
    IF = IsolationForest(n_estimators=100, bootstrap=True, contamination=contamination, random_state=0, n_jobs=-1)

### Start simulation.

    model.simulate(data, 
                   cal_ranges=[[0, 1000], [0, 100], [0, 100]],
                   n_cal_points=10000,
                   detector=IF, 
                   n_iterations=1000,
                   window_size=4*672,
                   filename="IF")
