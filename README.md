%%%%% Neutrino Distance Calculator %%%%%

A tool for analyzing gravitational wave skymap data and calculating distance estimates for neutrino events.

# Overview
This package calculates distance estimates for neutrino events by matching their coordinates with gravitational wave skymaps. It reads neutrino location data from a JSON file and GW distance information from a FITS file, then outputs an enhanced JSON file containing distance estimates for each neutrino.

# Features
Coordinate conversion from zenith/azimuth to RA/Dec
Multi-order HEALPix skymap processing
Distance probability distribution analysis
Calculation of distance mean and peak values for each neutrino
Probability mass calculation for each neutrino location

# Installation Requirements
- Python 3.6+
- NumPy
- Astropy
- Healpy
- LALSuite (optional, for some advanced functionality)
- LIGO Skymap utilities
- hpmoc (HEALPix Multi-Order Coverage)
