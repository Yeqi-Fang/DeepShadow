from astrotools import auger, coord, skymap
ncrs, emin = 3000, 18.5            # number of cosmic rays
lons = coord.rand_phi(ncrs)        # isotropic in phi (~Uniform(-pi, pi))
lats = coord.rand_theta(ncrs)      # isotropic in theta (Uniform in cos(theta))
vecs = coord.ang2vec(lons, lats)   # or better directly: coord.rand_vec(ncrs)
# Plot an example map with sampled energies. If you specify the opath keyword in
# the skymap function, the plot will be automatically saved and closed
log10e = auger.rand_energy_from_auger(n=ncrs, log10e_min=emin)
skymap.scatter(vecs, c=log10e, opath='isotropic_sky.png')


# https://astro.pages.rwth-aachen.de/astrotools/installation.html