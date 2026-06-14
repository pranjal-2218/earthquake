import requests
import pandas as pd

class USGSDataFetcher:
    URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"

    def fetch(self) -> pd.DataFrame:
        response = requests.get(self.URL)
        response.raise_for_status()
        data = response.json()

        records = []
        for feature in data["features"]:
            coords = feature["geometry"]["coordinates"]
            props = feature["properties"]

            if props["mag"] is not None:
                records.append({
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "depth": coords[2],
                    "magnitude": props["mag"],
                    "nst": props.get("nst"),
                    "tsunami": props.get("tsunami"),
                    "rms": props.get("rms"),
                    "gap": props.get("gap"),
                    "dmin": props.get("dmin")
                })

        df = pd.DataFrame(records)
        return df