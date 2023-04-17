Option Pricing Using Deep Learning: A combined Long Short-Term Memory and Multi-layer Perceptron Approach

This repository contains the implementation of a Master's thesis at NTNU that proposes a deep learning approach to price European call options on the S&P 500 index using a hybrid LSTM-MLP neural network.

Table of Contents
1. Requirements
2. Usage
3. Data
4. Contributing

Requirements
To run the code, you'll need the following software and packages:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- pandas
- matplotlib
- scikit-learn

Usage
1. Ensure that the dataset is placed in the `data` directory.
2. Run the `LSTM-MLP.ipynb` notebook. The training function is located within the notebook.

Data
You will need to create a `data` folder in the root directory of the project, as it is not included in the repository due to the .gitignore file.

The data can be obtained from the following sources:

- Option data: Download from https://www.optionsdx.com/
- Interest rate data: Download from https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2023

After obtaining the data, run the `data_processing.ipynb` notebook followed by the `data_filtering.ipynb` notebook. The resulting processed data should be placed in the `processed_data` folder, inside the data folder.

Contributing
If you'd like to contribute to the project, please feel free to submit pull requests or report issues.
