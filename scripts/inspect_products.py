"""
scripts/inspect_products.py
Inspect the official Copernicus Marine product catalogue and verify DOIs, dataset IDs,
variables, spatial and temporal coverage, and depth levels.
"""

import json
import os
import copernicusmarine

def inspect_copernicus_products():
    print("Fetching Copernicus Marine catalogue...")
    cat = copernicusmarine.describe()
    print(f"Total products in catalogue: {len(cat.products)}")

    target_dois = {
        "SST (OSTIA)": "10.48670/moi-00168",
        "SSS (Multi-Obs)": "10.48670/moi-00051",
        "SSH/SLA (DUACS)": "10.48670/moi-00145",
        "TARGET (GLORYS)": "10.48670/moi-00021"
    }

    results = {}

    for label, target_doi in target_dois.items():
        print(f"\n=======================================================")
        print(f"Searching for: {label} (DOI: {target_doi})")
        print(f"=======================================================")
        found = False
        for prod in cat.products:
            p_doi = prod.digital_object_identifier or ""
            if target_doi.lower() in p_doi.lower():
                found = True
                print(f"Product ID: {prod.product_id}")
                print(f"Title: {prod.title}")
                print(f"DOI: {prod.digital_object_identifier}")
                print(f"Processing level: {prod.processing_level}")
                print(f"Datasets ({len(prod.datasets)}):")
                
                prod_data = {
                    "product_id": prod.product_id,
                    "title": prod.title,
                    "doi": prod.digital_object_identifier,
                    "processing_level": prod.processing_level,
                    "datasets": []
                }

                for ds in prod.datasets:
                    ds_info = {
                        "dataset_id": ds.dataset_id,
                        "dataset_parts": []
                    }
                    print(f"  - Dataset ID: {ds.dataset_id}")
                    for part in getattr(ds, "dataset_parts", []):
                        for service in getattr(part, "services", []):
                            for var in getattr(service, "variables", []):
                                var_name = getattr(var, "short_name", "")
                                std_name = getattr(var, "standard_name", "")
                                units = getattr(var, "units", "")
                                coords = []
                                for coord in getattr(var, "coordinates", []):
                                    c_id = getattr(coord, "coordinate_id", "")
                                    c_min = getattr(coord, "minimum_value", "")
                                    c_max = getattr(coord, "maximum_value", "")
                                    c_step = getattr(coord, "step", "")
                                    c_vals = getattr(coord, "values", None)
                                    coords.append({
                                        "id": c_id,
                                        "min": c_min,
                                        "max": c_max,
                                        "step": c_step,
                                        "values_count": len(c_vals) if c_vals else None,
                                        "sample_values": c_vals[:5] if c_vals else None
                                    })
                                ds_info["dataset_parts"].append({
                                    "service_type": getattr(service, "service_type", ""),
                                    "variable": var_name,
                                    "standard_name": std_name,
                                    "units": units,
                                    "coordinates": coords
                                })
                    prod_data["datasets"].append(ds_info)
                results[label] = prod_data

        if not found:
            print(f"[WARN] No product found matching DOI {target_doi}")

    # Save to data/metadata/copernicus_metadata.json
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/copernicus_metadata.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nMetadata saved to data/metadata/copernicus_metadata.json")

if __name__ == "__main__":
    inspect_copernicus_products()
