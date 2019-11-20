## Simulate Data
Run "python Simulate.py" (This python file is based on the laidbax tutorial notebook -  https://github.com/XENON1T/laidbax/blob/master/notebooks/Tutorial.ipynb).

This will save two csv files called 'ERSIM.csv' and 'WIMPSIM.csv' in the data folder (~/pax/pax/data)	

"Simulate.py" can also be used to simulate data for other background particles, such as neutrons and neutrinos. 	

There are three different python codes for plotting the images included in this repository; 'Plotting_Peak.py', 'Plotting_Hit.py' and 'Plotting_HitPeak.py', which plot the three different types of images. In order to use one of these files, the file needs to be renamed to 'Plotting.py' and saved in the plotting folder in PaX (~/pax/pax/plugins/plotting). 

To run the simulated WIMP data through pax, in the command line write:
```
paxer --config XENON1T Simulation --input ~/pax/pax/data/WIMPSIM.csv --plot
```
Replace 'WIMPSIM' with 'ERSIM' and change line 107/107 in the plotting python file for the simulated ER data. 


## Running the CNN
To run the CNN create a conda environment containing TensorFlow:
```
conda create -n tensorflow_env tensorflow
  conda activate tensorflow_env
```
Then install "Keras", "Matplotlib", "Scikit-learn", "Pandas" and "Imageio":
  ```
  conda install -c conda-forge keras
  conda install -c conda-forge matplotlib
  conda install scikit-learn
  conda install -c anaconda pandas
  conda install -c menpo imageio
  ```
Then run CNN.py as normal.


#### Running TensorBoard
To run TensorBoard save the folders that are created during the running of the CNN into one folder.

Then run:
```
tensorboard --logdir=C:\<Name_of_folder>
```
