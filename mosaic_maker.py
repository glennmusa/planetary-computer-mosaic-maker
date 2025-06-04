# Copyright (c) 2025 Glenn Musa
# Portions generated with the assistance of GPT-4.1 (GitHub Copilot)
#
# See https://planetarycomputer.microsoft.com for more information and samples.
#
# Licensed under the MIT License. See LICENSE file in the project root for full
# license information.

import argparse
import logging
import os

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def cache_path(asset_url: str, cache_dir: str) -> str:
    """
    Returns a cache file path for a given asset URL.
    Uses the original filename for readability and stability.
    """
    filename = os.path.basename(asset_url.split("?")[0])
    return os.path.join(cache_dir, filename)


def download_asset(asset_url: str, cache_file: str) -> str:
    """
    Downloads the asset from the given URL to the cache file if not already
    cached.
    Returns the path to the cached file.
    """
    if not os.path.exists(cache_file):
        logging.info(f"Downloading {asset_url} to {cache_file}")
        import requests

        try:
            with requests.get(asset_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(cache_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            logging.error(f"Failed to download {asset_url}: {e}")
            raise
    else:
        logging.info(f"Cache hit for {cache_file}")
    return cache_file


def warp_band(src, bbox, width, height):
    """
    Warps and resamples a rasterio dataset to the target bbox and size.
    Returns a uint8 numpy array.
    """
    with WarpedVRT(
        src,
        crs="EPSG:4326",
        resampling=Resampling.bilinear,
        transform=rasterio.transform.from_bounds(*bbox, width, height),
        width=width,
        height=height,
    ) as vrt:
        arr = vrt.read(1, out_shape=(height, width))
        arr = np.clip(arr, 0, 10000)
        arr = (arr / 10000 * 255).astype(np.uint8)
        return arr


def validate_bbox(bbox):
    """
    Validates that the bounding box is within real-world geographic limits.
    """
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must have four values: "
            "[min_lon, min_lat, max_lon, max_lat]"
        )
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError(
            "Longitude values must be in [-180, 180] and min_lon < max_lon."
        )
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(
            "Latitude values must be in [-90, 90] and min_lat < max_lat."
        )
    return bbox


def compute_min_dimensions(bbox, min_width, min_height):
    """
    Computes (width, height) in pixels for a given bbox, ensuring dimensions
    are at least min_width and min_height, preserving the aspect ratio.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_width = max_lon - min_lon
    bbox_height = max_lat - min_lat
    aspect = bbox_width / bbox_height

    # Start with min_width, compute height
    width = min_width
    height = int(round(width / aspect))
    if height < min_height:
        # If height is too small, use min_height and compute width
        height = min_height
        width = int(round(height * aspect))
    return width, height


def compute_raw_dimensions(bbox, long_side=10980):
    """
    Computes (width, height) in pixels for the raw image,
    preserving aspect ratio,
    using the given long_side as the maximum dimension.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_width = max_lon - min_lon
    bbox_height = max_lat - min_lat
    aspect = bbox_width / bbox_height
    if aspect >= 1:
        width = long_side
        height = int(round(long_side / aspect))
    else:
        height = long_side
        width = int(round(long_side * aspect))
    return width, height


def search_catalog(bbox, date_range, cloud_cover, limit):
    """
    Search the Planetary Computer STAC API for Sentinel-2 L2A scenes.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        limit=limit,
    )
    items = list(search.items())
    logging.info(f"Found {len(items)} matching Sentinel-2 scenes.")
    return items


def process_item(
    item,
    bands,
    bbox,
    raw_width,
    raw_height,
    width,
    height,
    cache_dir,
    raw_mosaic,
    mosaic,
):
    """
    Download, warp, and update mosaics for a single item.
    """
    for i, band in enumerate(bands):
        asset = item.assets[band]
        cache_file = cache_path(asset.href, cache_dir)
        signed_url = planetary_computer.sign(asset.href)
        download_asset(signed_url, cache_file)
        try:
            with rasterio.open(cache_file) as src:
                arr_raw = warp_band(src, bbox, raw_width, raw_height)
                mask_raw = arr_raw > 0
                raw_mosaic[i][mask_raw] = np.maximum(
                    raw_mosaic[i][mask_raw], arr_raw[mask_raw]
                )

                arr_4k = warp_band(src, bbox, width, height)
                mask_4k = arr_4k > 0
                mosaic[i][mask_4k] = np.maximum(
                    mosaic[i][mask_4k], arr_4k[mask_4k]
                )
        except Exception as e:
            logging.error(f"Failed to read or warp {cache_file}: {e}")
            raise


def save_composite(mosaic, filename, ppi):
    """
    Save the composite image to disk.
    """
    rgb = np.stack(mosaic, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(filename, dpi=(ppi, ppi))
    logging.info(f"Saved {filename} at {ppi} PPI")


def main(
    bands,
    bbox,
    cache_dir,
    cloud_cover,
    date_range,
    limit,
    out_file,
    out_min_width,
    out_min_height,
    ppi,
    raw_file,
):
    """
    Orchestrates the workflow: search, process, and save composites.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    os.makedirs(cache_dir, exist_ok=True)
    bbox = validate_bbox(bbox)

    width, height = compute_min_dimensions(
        bbox, out_min_width, out_min_height
    )
    raw_width, raw_height = compute_raw_dimensions(bbox, long_side=10980)  
    logging.info(f"Output 4K image size: {width}x{height}")
    logging.info(f"Output raw image size: {raw_width}x{raw_height}")

    items = search_catalog(bbox, date_range, cloud_cover, limit)

    raw_mosaic = [
        np.zeros((raw_height, raw_width), dtype=np.uint8) for _ in bands
    ]
    mosaic = [
        np.zeros((height, width), dtype=np.uint8) for _ in bands
    ]

    for item_idx, item in enumerate(items, 1):
        logging.info(f"Processing item {item_idx}/{len(items)}: {item.id}")
        process_item(
            item,
            bands,
            bbox,
            raw_width,
            raw_height,
            width,
            height,
            cache_dir,
            raw_mosaic,
            mosaic,
        )

    save_composite(raw_mosaic, raw_file, ppi=72)
    save_composite(mosaic, out_file, ppi=ppi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and mosaic Sentinel-2 L2A imagery for any region"
        "saving both native and downsampled PNGs."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box as min_lon min_lat max_lon max_lat",
        default=[-113.2, 35.95, -111.9, 36.48],
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="satellite_cache",
        help="Directory to cache downloaded assets",
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=10,
        help="Maximum cloud cover percentage",
    )
    parser.add_argument(
        "--date-range",
        type=str,
        default="2017-12-01/2018-02-28",
        help="Date range for imagery",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of scenes to use"
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="composite_4k.png",
        help="Output filename for the downsampled PNG",
    )
    parser.add_argument(
        "--out-min-width",
        type=int,
        default=3840,
        help="Minimum width for the downsampled PNG")
    parser.add_argument(
        "--out-min-height",
        type=int,
        default=2160,
        help="Minimum height for the downsampled PNG")
    parser.add_argument(
        "--ppi",
        type=int,
        default=150,
        help="PPI (pixels per inch) for the downsampled PNG",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="composite_raw.png",
        help="Output filename for the native-resolution PNG",
    )

    args = parser.parse_args()

    main(
        bands=["B04", "B03", "B02"],
        bbox=args.bbox,
        cache_dir=args.cache_dir,
        cloud_cover=args.cloud_cover,
        date_range=args.date_range,
        limit=args.limit,
        out_file=args.out_file,
        out_min_height=args.out_min_height,
        out_min_width=args.out_min_width,
        ppi=args.ppi,
        raw_file=args.raw_file,
    )
