# planetary-computer-mosaic-maker

A Python tool hacked together to download, mosaic, and export true-color Sentinel-2 imagery from the Microsoft Planetary Computer for any region of interest.

## Usage

`mosaic_maker.py` will: 
 - search for cloud-free Sentinel-2 scenes using STAC
 - download and caches only the required bands
 - generate a native-resolution and a minimum-4K downsampled RGB PNG composite, preserving the geographic aspect ratio

```plaintext
python3 mosaic_maker.py --help
```

By default, the script will download an image of the Grand Canyon.
