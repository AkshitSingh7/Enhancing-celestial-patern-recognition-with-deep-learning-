import os
import galsim
import numpy as np
from astropy.table import Table
from astropy import wcs
from tqdm import tqdm

def generate_dataset_split(catalog_path, output_dir, num_images=100, img_size=512, fov_deg=1.0):
    """
    Generates synthetic images (.npy) and labels (.txt) for training/validation.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    print(f"Loading catalog for {output_dir}...")
    full_catalog = Table.read(catalog_path)
    
    # Create directories
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Simulation constants
    pixel_scale = fov_deg * 3600 / img_size
    rng = galsim.BaseDeviate(42)

    print(f"Generating {num_images} synthetic fields...")
    for i in tqdm(range(num_images)):
        # Randomly point telescope somewhere in the catalog density
        # For simplicity, we just pick a random star as the center
        center_star = full_catalog[np.random.randint(len(full_catalog))]
        ra_c, dec_c = center_star['ra'], center_star['dec']

        # WCS Setup
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [img_size / 2, img_size / 2]
        w.wcs.crval = [ra_c, dec_c]
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.cdelt = np.array([-fov_deg/img_size, fov_deg/img_size])

        # Filter stars in FOV
        half_fov = fov_deg / 2.0
        mask = (
            (full_catalog['ra'] > ra_c - half_fov) & (full_catalog['ra'] < ra_c + half_fov) &
            (full_catalog['dec'] > dec_c - half_fov) & (full_catalog['dec'] < dec_c + half_fov)
        )
        subset = full_catalog[mask]

        # Draw Image
        image = galsim.ImageF(img_size, img_size, scale=pixel_scale)
        labels = [] # List to store (class, x_norm, y_norm, w_norm, h_norm)

        for star in subset:
            ra, dec, mag = star['ra'], star['dec'], star['phot_g_mean_mag']
            flux = 10**(-0.4 * (mag - 25)) * 100  # Arbitrary flux scaling
            
            # Simple PSF
            psf = galsim.Gaussian(fwhm=1.0, flux=flux)
            
            # World -> Pixel
            world_pos = np.array([[ra, dec]])
            pix_pos = w.wcs_world2pix(world_pos, 0)[0]
            x_pix, y_pix = pix_pos

            if 0 <= x_pix < img_size and 0 <= y_pix < img_size:
                pos = galsim.PositionD(x_pix, y_pix)
                psf.drawImage(image=image, center=pos, add_to_image=True)
                
                # Create Label for this star
                # Format: class x_center y_center width height (normalized 0-1)
                # We assume a fixed small box size for point sources
                box_size = 10.0 / img_size
                labels.append(f"0 {x_pix/img_size:.6f} {y_pix/img_size:.6f} {box_size:.6f} {box_size:.6f}")

        # Add Noise
        image.addNoise(galsim.PoissonNoise(rng, sky_level=100.0))
        image.addNoise(galsim.GaussianNoise(rng, sigma=5.0))

        # Save Image as .npy (Dataset expects this)
        np.save(os.path.join(img_dir, f"field_{i:04d}.npy"), image.array)
        
        # Save Label as .txt
        with open(os.path.join(lbl_dir, f"field_{i:04d}.txt"), "w") as f:
            f.write("\n".join(labels))
