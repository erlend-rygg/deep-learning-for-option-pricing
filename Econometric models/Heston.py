import numpy as np
import pandas as pd
from numba import jit
from lmfit import Parameters, minimize
from scipy.integrate import quad
import time

i = complex (0,1) # Define complex number i

colab_path = "M:/Master/"

@jit
def charHeston(u, S0, r, T, sigma, kappa, theta, v0, rho):
    '''Implementation of the characteristic function of the Heston model'''
    # Frequent expression
    rsiu = rho*sigma*i*u

    # Calculate d
    d1 = (rsiu - kappa)**2
    d2 = sigma**2*(i*u + u**2)
    d = np.sqrt(d1 + d2)

    # Calculate g
    g1 = kappa - rsiu - d
    g2 = kappa - rsiu + d
    g = g1/g2

    #Calculate first exp
    exp1 = np.exp(r*T)

    #Calculate the first power
    base1 = S0
    exponent1 = i*u
    power1 = np.power(base1, exponent1)

    # Calculate second power
    base2 = (1-g*np.exp(-d*T)) / (1-g)
    exponent2 = -2*theta*kappa/(sigma**2)
    power2 = np.power(base2, exponent2)

    # Calculate the second exp
    part1 = theta*kappa*T/(sigma**2) * g1
    part2 = v0/(sigma**2) * g2 * (1 - np.exp(d*T))/(1 - g*np.exp(d*T))
    exp2 = np.exp(part1 + part2)

    # Main calculation
    return exp1*power1*power2*exp2

@jit
def integrand(u, S0, K, r, T, sigma, kappa, theta, v0, rho):
    '''Calculate the integrand of the Heston model'''
    numerator = np.exp(r*T)*charHeston(u-i, S0, r, T, sigma, kappa, theta, v0, rho) - K * charHeston(u, S0, r, T, sigma, kappa, theta, v0, rho)
    denominator = i*u *np.power(K, i*u)
    return np.real(numerator/denominator)

@jit(forceobj = True)
def priceHestonIntegral(S0, K, r, T, sigma, kappa, theta, v0, rho, maxIntegral = 100):
    '''Calculate integral for the price of a European call option using the Heston model'''
    integral = np.array([quad(integrand, 0, maxIntegral, args=(S0, K_i, r_i, T_i, sigma, kappa, theta, v0, rho))[0] for K_i, r_i, T_i in zip(K, r, T) ])
    return 0.5 * (S0 - K * np.exp(-r * T)) + integral/np.pi

def iter_cb(params, iter, resid):
    '''Callback function to print the parameters at each iteration of the minimizer for debuging'''
    parameters = [params['sigma'].value, 
                  params['kappa'].value, 
                  params['theta'].value, 
                  params['v0'].value, 
                  params['rho'].value, 
                  np.sum(resid)/len(resid)]
    print(parameters) 

def calibrateHeston(optionPrices, S0, strikes, rates, maturities, initial_parameters):
    '''Calibrate the Heston model parameters using the Levenberg Marquardt algorithm'''

    # Define the parameters to calibrate
    params = Parameters()
    params.add('sigma',value = initial_parameters[0], min = 1e-2, max = 1)
    params.add('kappa',value = initial_parameters[1], min = 1e-3, max = 5)
    params.add('theta',value = initial_parameters[2], min = 1e-4, max = 0.1)
    params.add('v0', value = initial_parameters[3], min = 1e-3, max = 0.5)
    params.add('rho', value = initial_parameters[4], min = -0.5, max = 0)

    # Define the objective function to minimize as squared errors
    objectiveFunctionHeston = lambda paramVect: (optionPrices - priceHestonIntegral(S0, strikes,  
                                                                        rates, 
                                                                        maturities, 
                                                                        paramVect['sigma'].value,                         
                                                                        paramVect['kappa'].value,
                                                                        paramVect['theta'].value,
                                                                        paramVect['v0'].value,
                                                                        paramVect['rho'].value)) **2   
    # Run the Levenberg Marquardt algorithm
    result = minimize(objectiveFunctionHeston, 
                      params, 
                      method = 'leastsq',
#                      iter_cb = iter_cb,
                      ftol = 1e-4) 
    return(result)

@jit(forceobj=True)
def create_data_np(df):
    '''Create numpy arrays with required data for calibration and testing'''
    optionPrices = df['Price'].values
    S0 = df['Underlying_last'].values[0]
    strikes = df['Strike'].values
    rates = df['R'].values
    maturities = df['TTM'].values

    data_np = np.empty((len(optionPrices), 5))
    data_np[:, 0] = optionPrices
    data_np[:, 1] = S0
    data_np[:, 2] = strikes
    data_np[:, 3] = rates
    data_np[:, 4] = maturities

    return data_np

def calculateHestonDate(data_cal, data_test, test_date, initial_parameters):
    '''Calculate the heston model parameters option data for date t-1 and calculate the option price for date t'''
    # Calibrate the Heston model
    calibrationResult = calibrateHeston(
        data_cal[:, 0],
        data_cal[0, 1],
        data_cal[:, 2],
        data_cal[:, 3],
        data_cal[:, 4],
        initial_parameters
    )

    # Extract the parameters
    params = np.array([
        calibrationResult.params['sigma'].value,
        calibrationResult.params['kappa'].value,
        calibrationResult.params['theta'].value,
        calibrationResult.params['v0'].value,
        calibrationResult.params['rho'].value
    ])

    # Price the options
    optionPricesHeston = priceHestonIntegral(
        data_test[0, 1],
        data_test[:, 2],
        data_test[:, 3],
        data_test[:, 4],
        *params
    )
    print(f'{(np.sum((optionPricesHeston - data_test[:, 0]) ** 2) / len(optionPricesHeston))**(0.5):.4f} RMSE for {test_date}')
    return optionPricesHeston, params

def HestonYear(df, year):
    '''Calculate the Heston model parameters for each day of the year and calculate the option price for the next day'''
    # Extract last date of previous year and apply yearly filter
    first_date = df[df['Quote_date'] < f'{year}-01-01']['Quote_date'].max()
    df_year = df[(df['Quote_date'] >= first_date) & (df['Quote_date'] <= f'{year}-12-31')]

    # Group the data by date extract numpy arrays with data for each day
    dates = np.sort(df_year['Quote_date'].unique())
    data_nps = df_year.groupby('Quote_date').apply(create_data_np)

    parameters = [0.02936084231134991, 0.040435318414851, 0.07357532840670486, 0.01697363646273281, -9.814464152491098e-08] # Initial parameters calculated on for last day of 2014
    optionPrices_list = []
    parameters_list = []
    
    t = time.time()
    for i in range(len(data_nps-1)):
        optionPrice, parameters = calculateHestonDate(data_nps[i], data_nps[i+1], dates[i+1], parameters)
        optionPrices_list.append(optionPrice)
        parameters_list.append(parameters)
    print(f'{year} finished with time: {time.time() - t}')  

    # Save option prices
    df = df[df['Quote_date'] != dates[0]]
    df['Heston_price'] = optionPrices_list
    print('=====================')
    print(f'Total RMSE {year}: {(np.sum((df["Heston_price"] - df["Price"]) ** 2) / len(df["Price"]))**(0.5)}')
    df.to_csv(f'./data/results/{dates[1]}_{dates[-1]} Heston.csv', index=False)

    # Save parameters
    df_params = pd.DataFrame(parameters_list, columns=['sigma', 'kappa', 'theta', 'v0', 'rho'])
    df_params['Quote_date'] = dates[1:]
    df_params.to_csv(f'./data/results/{dates[1]}_{dates[-1]} Heston parameters.csv', index=False)

if __name__ == '__main__':
    file = './data/processed_data/2014_2022_moneyness_and_TTM_filtered.csv'
    df = pd.read_csv(file)
    df['TTM'] = df['TTM'] / 365
    df['R'] = df['R'] / 100
    df.info()
    print(df.head(3))
    for year in range(2015, 2023):
        HestonYear(df, year)

    





