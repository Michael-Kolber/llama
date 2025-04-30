import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# — Standard library —
import json

# — Core 3rd-party —
import lal
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import norm

# — Astropy —
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable

# — LIGO skymap utilities —
from ligo.skymap.moc import uniq2pixarea

# — hpmoc (multi-order HEALPix) —
from hpmoc import PartialUniqSkymap
from hpmoc.points import PointsTuple

# — Astropy‐HEALPix —
from astropy_healpix import (
    uniq_to_level_ipix,
    level_to_nside,
    level_ipix_to_uniq
)

# — Project-specific utility —
from llama.files.i3.utils import zen_az2ra_dec

def compute_neutrino_distances(fits_filename, json_filepath, output_json_filename='lvc_skymap_distances.json', plot_width=600):
   
    # Open FITS file and set up header info
    hdulist = fits.open(fits_filename)
    
    # Load data from JSON file
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # Extract data into numpy arrays for coordinate conversion
    mjds = np.array([entry['mjd'] for entry in data])
    zens = np.array([entry['zenith'] for entry in data])
    azs = np.array([entry['azimuth'] for entry in data])

    # Perform the coordinate conversion: Zenith/Azimuth -> right ascension (RA)/declination (Dec)
    ra_deg, dec_deg = zen_az2ra_dec(mjds, zens, azs)

    # Creating the neutrino locations with only RA/Dec
    pts1 = PointsTuple([(ra, dec) for ra, dec in zip(ra_deg, dec_deg)])

    # Extract all points from the PointsTuple
    all_points = []
    for point_list in pts1:
        if point_list is not None:
            for point in point_list:
                all_points.append(point)

    # Extract RA/DEC from points
    ra_list = []
    dec_list = []
    for item in all_points:
        if isinstance(item, tuple):  
            ra, dec = item
            ra_list.append(ra)
            dec_list.append(dec)

    # Read skymap once
    skymap = QTable.read(fits_filename, hdu=1)
    uniq_arr = skymap['UNIQ']
    distmu_arr = skymap['DISTMU']
    prob_arr = skymap['PROBDENSITY']
    distmu_array = skymap['DISTMU']
    distsigma_array = skymap['DISTSIGMA']
    distnorm_array = skymap['DISTNORM']

    # Set up HEALPix ordering
    ordering = skymap.meta.get('ORDERING', 'NESTED')
    use_nested = ordering.upper() in ('NESTED','NUNIQ')

    # Build maps for quick lookups
    uniq_to_data = {uval: (dm, pr) for uval, dm, pr in zip(uniq_arr, distmu_arr, prob_arr)}
    uniq_to_row = {uval: idx for idx, uval in enumerate(uniq_arr)}
    levels = sorted({uniq_to_level_ipix(uval)[0] for uval in uniq_arr})

    # Function to query distmu and prob for a given RA/DEC
    def query_distmu_prob(ra_deg, dec_deg):
        θ = 0.5*np.pi - np.deg2rad(dec_deg)
        φ = np.deg2rad(ra_deg)

        for level in reversed(levels):
            nside = level_to_nside(level)
            ipix = hp.ang2pix(nside, θ, φ, nest=use_nested)
            uniq0 = level_ipix_to_uniq(level, ipix)
            if uniq0 in uniq_to_data:
                return uniq_to_data[uniq0], level, nside

        return (None, None), None, None

    # Process each neutrino position
    neutrino_dict = {}
    for idx, (ra, dec) in enumerate(zip(ra_list, dec_list), start=1):
        (distmu, prob), level, nside = query_distmu_prob(ra, dec)

        if level is None:
            print("  Position not covered by the MOC.")
            continue

        θ = 0.5*np.pi - np.deg2rad(dec)
        φ = np.deg2rad(ra)
        ipix = hp.ang2pix(nside, θ, φ, nest=use_nested)
        uniq0 = level_ipix_to_uniq(level, ipix)

        neutrino_dict[f"neutrino{idx}"] = {
            "theta": θ,
            "phi": φ,
            "ipix": ipix,
            "uniq": uniq0
        }

    # Process neutrinos to find distance distributions
    keys = sorted(neutrino_dict.keys(), key=lambda k: int(k.replace('neutrino','')))
    thetas = [neutrino_dict[k]['theta'] for k in keys]
    phis = [neutrino_dict[k]['phi'] for k in keys]
    ipixs = [neutrino_dict[k]['ipix'] for k in keys]
    uniqs = [neutrino_dict[k]['uniq'] for k in keys]

    # Set up distance array
    r = np.linspace(0, plot_width, plot_width)

    # Calculate distance distributions for each neutrino
    for idx, (theta, phi, ipix, uniq_index) in enumerate(
            zip(thetas, phis, ipixs, uniqs), start=1):

        level, _ = uniq_to_level_ipix(uniq_index)
        nside = level_to_nside(level)

        row = uniq_to_row[uniq_index]
        mu = distmu_array[row]
        sigma = distsigma_array[row]
        norm0 = distnorm_array[row]

        dp_dr = r**2 * norm0 * norm(mu, sigma).pdf(r)

        # Find peak of distribution
        peak_idx = np.argmax(dp_dr)
        r_peak = r[peak_idx]
        peak_value = dp_dr[peak_idx]
        
        # Store results in neutrino dictionary
        neutrino_dict[f"neutrino{idx}"]['mean'] = r_peak
        neutrino_dict[f"neutrino{idx}"]['std'] = peak_value
        
        print(f"\nNeutrino {idx}  –  peak @ {r_peak:.2f} Mpc  (value {peak_value:.3e})")
        print("ipix:       ", ipix)
        print("uniq index: ", uniq_index)

    # Calculate probability masses
    results = []
    for key, rec in neutrino_dict.items():
        uniq_idx = rec['uniq']
        row = uniq_to_row[uniq_idx]

        level, ipix = uniq_to_level_ipix(uniq_idx)
        nside = level_to_nside(level)
        probdens = skymap['PROBDENSITY'][row]
        pixarea = uniq2pixarea(uniq_idx)
        probmass = probdens * pixarea

        results.append((key, level, nside, probdens, pixarea, probmass))

    # Calculate total probability
    total_prob = sum(r[5] for r in results)
    results.sort(key=lambda x: x[5], reverse=True)

    # Print results table
    print("\n---------------------------------------------------------------------------------------------------------------")
    print("Neutrino #  | level | nside | probdens [1/sr] | pix area [sr] |    prob mass    | percent")
    print("            |       |       |                 |               |                 |(relative to other Neutrinos)")
    print("---------------------------------------------------------------------------------------------------------------")
    for key, level, nside, probdens, pixarea, probmass in results:
        percent = 100 * probmass / total_prob
        print(f"{key:10s}  {level:5d}   {nside:5d}   "
              f"{probdens:8.3e}      {pixarea:8.3e}      "
              f"{probmass:8.3e}   {percent:7.4f}%")
    print("---------------------------------------------------------------------------------------------------------------")

    # Get global distance information from header if available
    global_distmean = None
    global_diststd = None
    for i, h in enumerate(hdulist):
        if 'DISTMEAN' in h.header:
            global_distmean = h.header['DISTMEAN']
            global_diststd = h.header['DISTSTD']
            break

    # Create enhanced neutrino data with distance information
    enhanced_neutrinos = []

    # Add global distance values at the beginning of the JSON
    metadata = {
        "global_metadata": {
            "description": "GW skymap global distance information from FITS header",
            "DISTMEAN": float(global_distmean) if global_distmean is not None else None,
            "DISTSTD": float(global_diststd) if global_diststd is not None else None,
        }
    }
    enhanced_neutrinos.append(metadata)

    # Add each neutrino with its distance information
    for i, entry in enumerate(data):
        enhanced_entry = entry.copy()
        
        neutrino_key = f"neutrino{i+1}"
        if neutrino_key in neutrino_dict:
            # Get mean distance (r_peak) and standard deviation (peak_value)
            mean_value = neutrino_dict[neutrino_key].get('mean', 0)  # r_peak
            std_value = neutrino_dict[neutrino_key].get('std', 0)    # peak_value
            
            # Handle Astropy Quantity objects - extract value without units
            if hasattr(mean_value, 'value'):
                mean_value = mean_value.value
            if hasattr(std_value, 'value'):
                std_value = std_value.value
                
            enhanced_entry['distmean'] = float(mean_value)
            enhanced_entry['diststd'] = float(std_value)
        else:
            enhanced_entry['distmean'] = 0.0
            enhanced_entry['diststd'] = 0.0
        
        enhanced_neutrinos.append(enhanced_entry)

    # Save the enhanced neutrino data to a new JSON file
    with open(output_json_filename, 'w') as outfile:
        json.dump(enhanced_neutrinos, outfile, indent=4)

    print(f"\nEnhanced neutrino data saved to '{output_json_filename}'")

    hdulist.close()