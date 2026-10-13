import json
import math
from typing import Dict, List, Tuple

from datamodel import Order, TradingState


class Trader:
    PRODUCTS = ("AMBER", "BASALT", "COBALT", "DUNE", "EMBER")
    POSITION_LIMIT = 50
    SOFT_LIMIT = 35
    QUOTE_SIZE = 12
    MEMORY_VERSION = 1
    MAX_MEMORY = 50_000
    MODEL_PRIORS = (0.40, 0.20, 0.15, 0.15, 0.10)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _finite(value) -> bool:
        return (
            type(value) in (int, float)
            and abs(value) <= 1e12
            and math.isfinite(value)
        )

    @classmethod
    def _valid_record(cls, record) -> bool:
        if not isinstance(record, dict):
            return False
        if type(record.get("t")) is not int:
            return False
        if type(record.get("n")) is not int or not 1 <= record["n"] <= 1_000_000_000:
            return False
        if type(record.get("h")) is not int or record["h"] not in (0, 1):
            return False
        if not all(cls._finite(record.get(k)) for k in ("m", "a", "d", "v", "s")):
            return False
        if record["m"] <= 0 or record["a"] <= 0 or record["v"] < 0 or record["s"] < 0:
            return False
        for key in ("f", "l"):
            values = record.get(key)
            if not isinstance(values, list) or len(values) != 5:
                return False
            if not all(cls._finite(value) for value in values):
                return False
        return all(value >= 0 for value in record["l"])

    @classmethod
    def _load_memory(cls, raw: str) -> dict:
        if not isinstance(raw, str) or not raw or len(raw) > cls.MAX_MEMORY:
            return {}
        try:
            data = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            return {}
        if not isinstance(data, dict) or data.get("version") != cls.MEMORY_VERSION:
            return {}
        records = data.get("products")
        if not isinstance(records, dict):
            return {}
        return {
            product: {key: records[product][key]
                      for key in ("t", "n", "h", "m", "a", "d", "v", "s", "f", "l")}
            for product in cls.PRODUCTS
            if product in records and cls._valid_record(records[product])
        }

    @staticmethod
    def _book(depth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(
            ((p, q) for p, q in depth.buy_orders.items()
             if type(p) is int and p > 0 and type(q) is int and q > 0),
            reverse=True,
        )
        asks = sorted(
            ((p, -q) for p, q in depth.sell_orders.items()
             if type(p) is int and p > 0 and type(q) is int and q < 0)
        )
        return bids, asks

    @classmethod
    def _weights(cls, record: dict) -> List[float]:
        scores = [prior / (0.25 + loss)
                  for prior, loss in zip(cls.MODEL_PRIORS, record["l"])]
        total = sum(scores)
        learned = [score / total for score in scores]
        confidence = min(1.0, max(0.0, (record["n"] - 1) / 40.0))
        return [
            (1.0 - confidence) * prior
            + confidence * (0.10 * prior + 0.90 * weight)
            for prior, weight in zip(cls.MODEL_PRIORS, learned)
        ]

    @classmethod
    def _estimate(cls, bids, asks, previous, timestamp):
        bid, bid_size = bids[0]
        ask, ask_size = asks[0]
        mid = (bid + ask) / 2.0
        spread = float(ask - bid)
        shock = False

        if previous is None or timestamp < previous["t"]:
            record = {
                "t": timestamp, "n": 1, "h": 0, "m": mid, "a": mid, "d": 0.0,
                "v": 0.25, "s": spread,
                "f": [mid] * 5, "l": [max(1.0, spread * spread / 4.0)] * 5,
            }
        elif timestamp == previous["t"]:
            weights = cls._weights(previous)
            fair = sum(w * f for w, f in zip(weights, previous["f"]))
            return fair, math.sqrt(previous["v"]), previous, bool(previous["h"])
        else:
            record = dict(previous)
            delta = mid - previous["m"]
            old_sigma = math.sqrt(previous["v"])
            shock = abs(delta) > max(8.0, 6.0 * old_sigma, 3.0 * previous["s"])
            loss_cap = max(25.0, 25.0 * previous["v"], 4.0 * spread * spread)
            record["l"] = [
                0.97 * loss + 0.03 * min((mid - forecast) ** 2, loss_cap)
                for loss, forecast in zip(previous["l"], previous["f"])
            ]
            record["n"] = min(previous["n"] + 1, 1_000_000_000)
            record["t"] = timestamp
            record["h"] = int(shock)
            record["m"] = mid
            record["s"] = 0.95 * previous["s"] + 0.05 * spread
            variance_cap = max(100.0, 36.0 * previous["v"], 9.0 * spread * spread)
            record["v"] = max(
                0.04, 0.95 * previous["v"] + 0.05 * min(delta * delta, variance_cap)
            )
            if shock:
                record["a"] = mid
                record["d"] = 0.0
                record["l"] = [max(1.0, record["v"])] * 5
            else:
                record["a"] = 0.96 * previous["a"] + 0.04 * mid
                record["d"] = 0.88 * previous["d"] + 0.12 * delta

        sigma = math.sqrt(record["v"])
        queue_shift = (spread / 2.0) * (bid_size - ask_size) / (bid_size + ask_size)
        near_bids, near_asks = bids[:3], asks[:3]
        bid_vwap = sum(p * q for p, q in near_bids) / sum(q for _, q in near_bids)
        ask_vwap = sum(p * q for p, q in near_asks) / sum(q for _, q in near_asks)
        depth_center = (bid_vwap + ask_vwap) / 2.0
        band = max(2.0, 2.0 * spread, 3.0 * sigma)
        record["f"] = [
            mid,
            mid + queue_shift,
            cls._clip(depth_center, mid - band, mid + band),
            cls._clip(record["a"], mid - band, mid + band),
            mid + cls._clip(record["d"], -max(1.0, sigma), max(1.0, sigma)),
        ]
        weights = cls._weights(record)
        fair = sum(w * f for w, f in zip(weights, record["f"]))
        return fair, sigma, record, shock

    @classmethod
    def _inventory_penalty(cls, position: int, risk_scale: float) -> float:
        fraction = position / cls.POSITION_LIMIT
        return risk_scale * (fraction + 1.5 * fraction ** 3)

    @classmethod
    def _orders(cls, symbol, bids, asks, position, fair, sigma, shock):
        orders = []
        buy_room = cls.POSITION_LIMIT - position
        sell_room = cls.POSITION_LIMIT + position
        bought = sold = 0
        spread = asks[0][0] - bids[0][0]
        risk_scale = max(0.60, 0.40 * spread, sigma)
        take_edge = 0.50 + 0.25 * sigma

        for price, available in asks:
            quantity = 0
            while quantity < available and quantity < buy_room:
                next_position = position + bought + quantity + 1
                reservation = fair - cls._inventory_penalty(next_position, risk_scale)
                if price + take_edge > reservation:
                    break
                if shock and abs(next_position) >= abs(next_position - 1):
                    break
                quantity += 1
            if quantity:
                orders.append(Order(symbol, int(price), int(quantity)))
                buy_room -= quantity
                bought += quantity
            if quantity < available or buy_room == 0:
                break

        for price, available in bids:
            quantity = 0
            while quantity < available and quantity < sell_room:
                next_position = position - sold - quantity - 1
                reservation = fair - cls._inventory_penalty(next_position, risk_scale)
                if price - take_edge < reservation:
                    break
                if shock and abs(next_position) >= abs(next_position + 1):
                    break
                quantity += 1
            if quantity:
                orders.append(Order(symbol, int(price), -int(quantity)))
                sell_room -= quantity
                sold += quantity
            if quantity < available or sell_room == 0:
                break

        if not shock:
            half_width = max(0.50, 0.15 * spread, 0.50 * sigma)
            buy_position = position + bought
            sell_position = position - sold
            buy_fair = fair - cls._inventory_penalty(buy_position, risk_scale)
            sell_fair = fair - cls._inventory_penalty(sell_position, risk_scale)
            bid_price = min(
                math.floor(buy_fair - half_width), bids[0][0] + 1, asks[0][0] - 1
            )
            ask_price = max(
                math.ceil(sell_fair + half_width), asks[0][0] - 1, bids[0][0] + 1
            )
            buy_quantity = min(cls.QUOTE_SIZE, buy_room, max(0, cls.SOFT_LIMIT - buy_position))
            sell_quantity = min(cls.QUOTE_SIZE, sell_room, max(0, cls.SOFT_LIMIT + sell_position))
            if bid_price > 0 and buy_quantity > 0 and bid_price < ask_price:
                orders.append(Order(symbol, int(bid_price), int(buy_quantity)))
            if ask_price > 0 and sell_quantity > 0 and bid_price < ask_price:
                orders.append(Order(symbol, int(ask_price), -int(sell_quantity)))

        buys = sum(order.quantity for order in orders if order.quantity > 0)
        sells = -sum(order.quantity for order in orders if order.quantity < 0)
        if position + buys > cls.POSITION_LIMIT or position - sells < -cls.POSITION_LIMIT:
            return []
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = self._load_memory(getattr(state, "traderData", ""))
        timestamp = int(state.timestamp)
        result = {}

        for symbol in self.PRODUCTS:
            result[symbol] = []
            depth = state.order_depths.get(symbol)
            if depth is None:
                continue
            position = state.position.get(symbol, 0)
            if type(position) is not int or abs(position) > self.POSITION_LIMIT:
                continue
            bids, asks = self._book(depth)
            if not bids or not asks or bids[0][0] >= asks[0][0]:
                continue
            fair, sigma, record, shock = self._estimate(
                bids, asks, memory.get(symbol), timestamp
            )
            memory[symbol] = record
            result[symbol] = self._orders(
                symbol, bids, asks, position, fair, sigma, shock
            )

        trader_data = json.dumps(
            {"version": self.MEMORY_VERSION, "products": memory},
            separators=(",", ":"), allow_nan=False,
        )
        if len(trader_data) > self.MAX_MEMORY:
            trader_data = '{"version":1,"products":{}}'
        return result, 0, trader_data
