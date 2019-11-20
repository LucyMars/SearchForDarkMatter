# SearchForDarkMatter

This is GitHub repository includes all the code used for our paper "Convolutional Neural Networks for Direct Detection of Dark Matter".

This project was based on the XENON1T experiment by the XENON collaboration and used the open sourced software that they developed to simulate proposed WIMP (Weakly Interacting Massive Particle) and background events.

To run this project, PaX, Laidbax, Blueice and Multhirst need to be installed - see "Installation.md".

Simulate.py is used to simulate WIMP and electronic recoil (ER) background events and save the data in a csv file ("WIMP.csv" or "ER.csv").

The csv files are then put through PaX (Processor for Analysing XENON) to produce a graphical representation of the event. We used these graphical images in a convolutional neural network (CNN) to train the model to identify the difference between background and signal.

We tested three images:

1. "Hit" - The hitpattern of the photomultiplier tubes (PMTs) in the time projection chamber (TPC) (Plotting_Hit.py)
2. "Peak" - The largest S1 and S2 peaks (Plotting_Peak.py)
3. "HitPeak" - An image that shows both hitpattern and peak data (Plotting_HitPeak.py)

These three python files need to be saved in the pax plotting folder "~pax\pax\plugins\plotting" (see "Installation.md" to install PaX). To generate the images, the name of the file (e.g. Plotting_Hit.py) needs to be changed to Plotting.py.

The main CNN we used is shown in CNN.py.

Simulate.md describes how to simulate the data and run the CNN.
