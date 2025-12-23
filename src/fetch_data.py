import os
from astroquery.gaia import Gaia

def get_gaia_catalog(output_file, limit=None, max_mag=15):
    """
    Queries Gaia DR3 for stars brighter than max_mag and saves to FITS.
    """
    if os.path.exists(output_file):
        print(f"Catalog found at {output_file}, skipping download.")
        return

    print("Connecting to Gaia Archive...")
    
    # Construct ADQL Query
    limit_clause = f"TOP {limit}" if limit else ""
    query = f"""
    SELECT {limit_clause}
        source_id, ra, dec, phot_g_mean_mag
    FROM
        gaiadr3.gaia_source
    WHERE
        phot_g_mean_mag < {max_mag}
    """
    
    print(f"Launching query: {query}")
    job = Gaia.launch_job_async(query, dump_to_file=False)
    results = job.get_results()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving {len(results)} stars to {output_file}...")
    results.write(output_file, format='fits', overwrite=True)
