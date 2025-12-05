"""
Script for downloading Binance historical data to local storage.
"""

import argparse
import asyncio
import io
import os
import time
import zipfile
from typing import Any, Dict, List, Set

import aiohttp
import aiohttp_socks
import numpy as np
import pandas as pd
import zipfile
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from urllib.parse import quote_plus


class BinanceHistoryDataDL:
    """
    Download Binance historical data.

    Currently supports only USDT-margined perpetual contracts.
    """

    BASE_URL: str = "https://data.binance.vision/"

    BASE_SCRAPE_URL: str = "https://s3-ap-northeast-1.amazonaws.com/" \
        "data.binance.vision?delimiter=/&prefix="

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self.session: aiohttp.ClientSession = session

    def get_suffix_url(
        self,
        data_type: str,
        symbol: str | None = None,
        file: str | None = None,
    ) -> str:
        """
        Return the suffix URL based on data type, symbol, and file name.
        """
        url: str = "data/futures/um/daily/"
        if data_type == "trade":
            url += "trades/"
        elif data_type == "orderbook":
            url += "bookTicker/"
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        if symbol is not None:
            url += f"{symbol}/"

        if file is not None:
            url += file
        return url

    async def scrape_symbols(self, data_type: str) -> List[str]:
        """
        Fetch all available trading symbols.
        """
        part_scrape_url: str = (
            self.BASE_SCRAPE_URL + self.get_suffix_url(data_type)
        )
        is_truncated: bool = True
        next_marker: str = ""
        symbols: List[str] = []

        while True:
            scrape_url: str = part_scrape_url
            if len(next_marker) > 0:
                scrape_url = scrape_url + f"&marker={quote_plus(next_marker)}"
            
            # Parse symbol names from the URL.
            async with self.session.get(scrape_url) as resp:
                resp.raise_for_status()
                xml_bytes = await resp.read()
            soup = BeautifulSoup(xml_bytes, "xml")
            symbols.extend([
                key.text for key in soup.find_all("CommonPrefixes")
            ])

            # Check whether pagination is required.
            is_truncated = eval(soup.find("IsTruncated").text.capitalize())
            if is_truncated:
                next_marker = soup.find("NextMarker").text
            if not is_truncated:
                break

        # Remove URL prefixes and symbols that are not of interest.
        symbols = [symbol[: -1].split('/')[-1] for symbol in symbols]
        symbols = [symbol for symbol in symbols if symbol[-4: ] == "USDT"]
        return symbols

    async def scrape_files_url(self, data_type: str, symbol: str) -> List[str]:
        """
        Fetch all downloadable file URLs for a given symbol.
        """
        part_scrape_url: str = (
            self.BASE_SCRAPE_URL + self.get_suffix_url(data_type, symbol)
        )
        is_truncated: bool = True
        next_marker: str = ""
        file_urls: List[str] = []

        while True:
            scrape_url: str = part_scrape_url
            if len(next_marker) > 0:
                scrape_url = scrape_url + f"&marker={quote_plus(next_marker)}"
            
            # Parse file URLs from the URL.
            async with self.session.get(scrape_url) as resp:
                resp.raise_for_status()
                xml_bytes = await resp.read()
            soup = BeautifulSoup(xml_bytes, "xml")
            file_urls.extend([
                key.text for key in soup.find_all("Key")
            ])

            # Check whether pagination is required.
            is_truncated = eval(soup.find("IsTruncated").text.capitalize())
            if is_truncated:
                next_marker = soup.find("NextMarker").text
            if not is_truncated:
                break

        # Keep only ZIP files.
        file_urls = [url for url in file_urls if url[-4: ] == ".zip"]
        return file_urls

    async def download_trade(self, url: str) -> pd.DataFrame:
        """
        Download a trade file.
        """
        df: pd.DataFrame = await self._download_file(
            url,
            ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
            ["id", "price", "volume", "amount", "timestamp", "is_buyer_maker"],
            [str, float, float, float, float, str],
        )
        df["is_buyer_maker"] = df["is_buyer_maker"].str.lower().map(
            {"true": True, "false": False}
        ).astype(bool)
        return df

    async def download_orderbook(self, url: str) -> pd.DataFrame:
        """
        Download an order book file.
        """
        return await self._download_file(
            url,
            [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time"
            ],
            [
                "id",
                "bid_price1",
                "bid_amount1",
                "ask_price1",
                "ask_amount1",
                None,
                "timestamp",
            ],
            [str, float, float, float, float, None, float],
        )

    async def _download_file(
        self,
        url: str,
        raw_cols: List[str],
        select_cols: List[str | None],
        dtypes: List[str | None],
    ) -> pd.DataFrame:
        """
        Download a DataFrame from a file URL.
        """
        assert len(raw_cols) == len(select_cols) == len(dtypes)
        select_col_ids: List[int] = [
            i for i, col in enumerate(select_cols) if col is not None
        ]
        select_cols = [col for col in select_cols if col is not None]
        dtypes = [dtype for dtype in dtypes if dtype is not None]

        async with self.session.get(self.BASE_URL + url) as resp:
            resp.raise_for_status()
            zip_bytes = await resp.read()

        # Extract and read the CSV file.
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open(zf.namelist()[0], 'r') as csv_file:
                df = pd.read_csv(
                    csv_file,
                    usecols=select_col_ids,
                    header=None,
                    low_memory=False,
                )

        # Check whether the first row is actually the header.
        first_row: List[Any] = df.iloc[0].tolist()
        if any(i in first_row for i in raw_cols):
            df = df.iloc[1: ]

        # Adjust column names and dtypes.
        df.columns = select_cols
        df = df.astype(dict(zip(select_cols, dtypes)))
        return df.set_index("id")


async def main(args: argparse.Namespace) -> None:
    assert os.path.exists(args.save_folder), f"No exist {args.save_folder}"
    os.makedirs(os.path.join(args.save_folder, "trade"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "orderbook"), exist_ok=True)

    connector = aiohttp_socks.ProxyConnector.from_url(args.proxy_url)
    async with aiohttp.ClientSession(connector=connector) as session:
        downloader = BinanceHistoryDataDL(session)

        # Update trades.
        symbols: List[str] = await downloader.scrape_symbols("trade")
        for symbol in tqdm(symbols, desc="update trades"):
            folder: str = os.path.join(args.save_folder, "trade", symbol)
            os.makedirs(folder, exist_ok=True)

            # Compute the dates that need to be updated.
            exist_dts: Set[np.datetime64] = set(
                np.datetime64(file[-18: -8], 'D') for file in os.listdir(folder)
            )

            file_urls: List[str] = await downloader.scrape_files_url(
                "trade", symbol
            )
            file_dts: Dict[np.datetime64, str] = {
                np.datetime64(url[-14: -4], 'D'): url for url in file_urls
            }
            file_dts = {
                dt: file_dts[dt] for dt in file_dts
                if dt not in exist_dts and dt >= np.datetime64(
                    args.start_dt, 'D'
                )
            }

            # Download and save files.
            for dt in tqdm(file_dts, desc=f"update {symbol} trades"):
                df: pd.DataFrame = await downloader.download_trade(file_dts[dt])
                df.to_feather(os.path.join(folder, f"{dt.astype(str)}.feather"))

        # # Update order books.
        # symbols: List[str] = await downloader.scrape_symbols("orderbook")
        # for symbol in tqdm(symbols, desc="update orderbooks"):
        #     folder: str = os.path.join(args.save_folder, "orderbook", symbol)
        #     os.makedirs(folder, exist_ok=True)

        #     # Compute the dates that need to be updated.
        #     exist_dts: Set[np.datetime64] = set(
        #         np.datetime64(file[-18: -8], 'D') for file in os.listdir(folder)
        #     )

        #     file_urls: List[str] = await downloader.scrape_files_url(
        #         "orderbook", symbol
        #     )
        #     file_dts: Dict[np.datetime64, str] = {
        #         np.datetime64(url[-14: -4], 'D'): url for url in file_urls
        #     }
        #     file_dts = {
        #         dt: file_dts[dt] for dt in file_dts
        #         if dt not in exist_dts and dt >= np.datetime64(
        #             args.start_dt, 'D'
        #         )
        #     }

        #     # Download and save files.
        #     for dt in tqdm(file_dts, desc=f"update {symbol} orderbooks"):
        #         df: pd.DataFrame = await downloader.download_orderbook(file_dts[dt])
        #         df.to_feather(os.path.join(folder, f"{dt.astype(str)}.feather"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--start_dt", type=str, default="2020-01-01")
    # parser.add_argument("--end_dt", type=str, default=None)
    parser.add_argument("--proxy_url", type=str, default="socks5://localhost:11889")
    args: argparse.Namespace = parser.parse_args()

    attempt: int = 0
    while True:
        attempt += 1
        try:
            print("start update, attempt:", attempt)
            asyncio.run(main(args))
            print("update done")
            break
        except (asyncio.TimeoutError, TimeoutError) as e:
            print("catch TimeoutError, wait 60 seconds and retry")
            time.sleep(60)
