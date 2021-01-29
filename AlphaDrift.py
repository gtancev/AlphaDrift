__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class Sensor:
    """
    Sensor class that specifies sensitivities, zero current,
    and drift of sensitivity and zero current.

    Inputs:
    s = array of upper and lower bounds for all sensitivities in nA/ppb
    z = array of pper and lower bound for zero current in nA
    d_s = array of upper and lower bound of drifts in 1-%change/(year*100)
    d_z = array of upper and lower bound of zero drift in nA/year
    c = array of positions with same properties,
        e.g., combined sensors OX-B4 has same s and d_s for NO2 and O3

    Outputs:
    Instance of class sensor.
    """
    def __init__(self, s=[[0.42, 0.65], [0.0, 0.0], [0.0, 0.0]],
                 z=[-250.0, 30.0],
                 d_s=[[0.9, 1.0], [1.0, 1.0], [1.0, 1.0]],
                 d_z=[-50.0, 50.0],
                 c=[]):
        self.s = np.asarray(s)
        self.z = np.asarray(z)
        self.d_s = np.asarray(d_s)
        self.d_z = np.asarray(d_z)
        self.c = np.asarray(c)

        # this piece checks that s and d_s are the same
        if len(c):
            ref = c[0]
            for idx in c[1:]:
                assert s[idx] == s[ref]
                assert d_s[idx] == d_s[ref]


class AlphaDrift:
    """
    Class for Monte Carlo simulation of sensor drift.

    Inputs:
    List of sensors of class sensor.

    Outputs:
    Instance of class AlphaDrift.
    """
    def __init__(self, sensors=[]):
        if not len(sensors):
            raise Exception("Provide some sensors.")
        else:
            self.sensors = sensors

    @property
    def n_sensors(self):
        """
        Count the number of sensors.

        Input:
        Instance of AlphaDrift.

        Output:
        Number of sensors.
        """
        return len(self.sensors)

    @staticmethod
    def moving_average(x, w):
        """
        Function to compute moving averages.

        Input:
        x = array of any shape
        w = window size

        Output:
        array of the same shape as x
        """
        return np.divide(np.convolve(x, np.ones(w), "same"), w)

    @staticmethod
    def sample_uniform(low, high, size):
        """
        Generates uniformly distributed numbers
        in the interval between low and high.
        """
        return np.random.uniform(low, high, size)

    @staticmethod
    def sample_normal(loc, scale, size):
        """
        Generates normally distributed numbers
        in the interval between low and high.
        """
        return np.random.normal(loc, scale, size)

    def _manufacture_sensors(self, n_factors):
        """
        Create random model for all sensors.

        Input:
        Instance of AlphaDrift.

        Output:
        S = sensitivity matrix of shape (n_sensors, n_factors)
        Z = zero vector of shape (n_sensors, 1)
        D_S = sensitivity drift matrix of shape (n_sensors, n_factors)
        D_Z = zero drift of shape (n_sensors, 1)
        """
        n_sensors = self.n_sensors

        S = np.zeros((n_sensors, n_factors))
        Z = np.zeros((n_sensors, 1))
        D_S = np.zeros((n_sensors, n_factors))
        D_Z = np.zeros((n_sensors, 1))

        for i, sensor in enumerate(self.sensors):
            s_lbs, s_ubs = sensor.s[:, 0], sensor.s[:, 1]
            S[i, :] = self.sample_uniform(s_lbs, s_ubs, n_factors)

            z_lbs, z_ubs = sensor.z[0], sensor.z[1]
            Z[i, 0] = self.sample_uniform(z_lbs, z_ubs, 1)

            ds_lbs, ds_ubs = sensor.d_s[:, 0], sensor.d_s[:, 1]
            D_S[i, :] = self.sample_uniform(ds_lbs, ds_ubs, n_factors)

            dz_lbs, dz_ubs = sensor.d_z[0], sensor.d_z[1]
            D_Z[i, 0] = self.sample_uniform(dz_lbs, dz_ubs, 1)

            # account for combined sensor
            if len(sensor.c):
                c = sensor.c[0]
                for k in sensor.c[1:]:
                    S[i, k], D_S[i, k] = S[i, c], D_S[i, c]

        return S, Z, D_S, D_Z

    def _generate_calibration(self, S, Z,
                              cal_ranges, n_factors,
                              n_cal_points=1000,
                              noise=2.0):
        """
        Calibration of sensors. Expose sensors to conditions
        defined by calibration ranges. Sensor response is obtained.

        Input:
        S = sensitivity matrix of shape (n_sensor, n_factors)
        Z = zero matrix of shape (n_sensor, 1)
        n_factors = number of gases
        n_cal_points = number of calibration points
        noise = noise in signal (standard deviation)

        Output:
        Y = signal matrix of shape (n_points, n_sensors)
        """
        cal_ranges = np.asarray(cal_ranges)
        r_lbs, r_ubs = cal_ranges[:, 0], cal_ranges[:, 1]

        X = self.sample_uniform(r_lbs, r_ubs, (n_cal_points, n_factors))

        Y = np.add(np.dot(S, np.transpose(X)), Z)
        e = self.sample_normal(0.0, noise, Y.shape)

        return np.transpose(np.add(Y, e))

    def _generate_drift(self, S, Z, D_S, D_Z,
                        n_points_per_hour, n_time_points):
        """
        Computation of sensitivities and zero lines for each time point.

        Input:
        S = sensitivity matrix of shape (n_sensors, n_factors)
        D_S = sensitivity drift matrix of shape (n_sensors, n_factors)
        Z = zero of shape (n_sensors, 1)
        D_Z = zero drift of shape (n_sensors, 1)
        n_points_per_hour = samples per hour
        n_time_points = number of time points
        """
        t = np.expand_dims(np.arange(0, n_time_points, 1), 0)
        n_points_per_year = n_points_per_hour*24*365

        D_S_m = np.power(D_S, 1 / n_points_per_year)
        S_t_ = np.power(np.expand_dims(D_S_m, -1), t)
        S_ = np.repeat(np.expand_dims(S, axis=-1), n_time_points, axis=-1)
        S_t = np.multiply(S_, S_t_)

        D_Z_m = np.divide(D_Z, n_points_per_year)
        Z_t_ = np.multiply(D_Z_m, t)
        Z_t = np.add(Z_t_, Z)

        return S_t, Z_t

    def _compute_signals(self, S_t, Z_t, data, noise=2.0):
        """
        Function that compute the signals over time.

        Input:
        S_t = sensitivity matrix of shape (n_sensors, n_factors, n_timepoints)
        Z_t = zero vector of shape (n_sensors, n_time_points)
        data = data of shape (n_factors, n_time_points)
        noise = noise in signal (standard deviation)

        Output:
        Y = sensor signals of shape (n_time_points, n_sensors)
        """
        Y = np.add(np.sum(np.multiply(np.transpose(data), S_t), axis=1), Z_t)
        e = self.sample_normal(0.0, noise, Y.shape)

        return np.transpose(np.add(Y, e))

    def simulate(self, data, n_points_per_hour=4,
                 cal_ranges=[[0, 1000], [0, 100], [0, 100]],
                 n_cal_points=1000,
                 scaler=MinMaxScaler(),
                 detector=IsolationForest(n_estimators=100,
                                          contamination="auto",
                                          n_jobs=-1),
                 n_iterations=1000,
                 window_size=672,
                 filename="results"):
        """
        Method to call for simulation.

        Input:
        data = array of shape (n_data_points, n_factors)
        n_points_per_hour = amount of samples per hour
        cal_ranges = ranges for factors to calibrate of shape (n_factors, 2)
        n_cal_points = number of points for calibration
        detector = instance of anomaly detector
        n_iterations = number of simulation iterations
        window_size = window for moving average
        filename = how to name the file with results

        Output:
        file of name results with simulation content
        """

        n_time_points, n_factors = data.shape[0], data.shape[1]
        w = window_size

        results = np.zeros((n_iterations, n_time_points))

        for k in range(n_iterations):

            S, Z, D_S, D_Z = self._manufacture_sensors(n_factors)

            Y_cal = self._generate_calibration(S, Z,
                                               cal_ranges,
                                               n_factors,
                                               n_cal_points)

            Y_cal_s = scaler.fit_transform(Y_cal)

            detector.fit(Y_cal_s)

            S_t, Z_t = self._generate_drift(S, Z, D_S, D_Z,
                                            n_points_per_hour,
                                            n_time_points)

            Y = self._compute_signals(S_t, Z_t, data)

            Y_s = scaler.transform(Y)

            is_outlier = np.divide(np.add(detector.predict(Y_s), -1), -2)

            results[k, :] = self.moving_average(is_outlier, w)

        np.save(filename, results)

        return None
