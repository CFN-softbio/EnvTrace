from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, List, Dict

@dataclass(frozen=True)
class Event:
    channel: str
    timestamp: float
    value: Any
    meta: Dict[str, Any] = field(default_factory=dict)

class Trace:
    def __init__(self, events: Iterable[Event] | None = None) -> None:
        self.events: List[Event] = list(events or [])

    def sort_by_time(self, in_place: bool = True) -> "Trace":
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)
        if in_place:
            self.events = sorted_events
            return self
        return Trace(sorted_events)

    def zero_time(self, zero_at: Optional[float] = None, in_place: bool = True) -> "Trace":
        if not self.events:
            return self if in_place else Trace([])
        if zero_at is None:
            zero_at = self.events[0].timestamp
        adjusted = [
            Event(e.channel, float(e.timestamp) - float(zero_at), e.value, dict(e.meta))
            for e in self.events
        ]
        if in_place:
            self.events = adjusted
            return self
        return Trace(adjusted)

    def filter_by_channels(self, channels: Iterable[str]) -> "Trace":
        s = set(channels)
        return Trace([e for e in self.events if e.channel in s])

    def filter_by_time_window(self, start: Optional[float] = None, end: Optional[float] = None) -> "Trace":
        return Trace([
            e for e in self.events
            if (start is None or e.timestamp >= start) and (end is None or e.timestamp <= end)
        ])

    def canonicalize(self, alias_map: Dict[str, str]) -> "Trace":
        return Trace([
            Event(alias_map.get(e.channel, e.channel), e.timestamp, e.value, dict(e.meta))
            for e in self.events
        ])

    def deduplicate(self, coalesce_window: Optional[float] = None) -> "Trace":
        if coalesce_window is None or not self.events:
            return Trace(self.events.copy())
        coalesced: List[Event] = []
        for e in self.events:
            if (
                coalesced
                and e.channel == coalesced[-1].channel
                and abs(e.timestamp - coalesced[-1].timestamp) <= coalesce_window
            ):
                coalesced[-1] = Event(e.channel, e.timestamp, e.value, dict(e.meta))
            else:
                coalesced.append(e)
        return Trace(coalesced)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1",
            "events": [self._event_to_dict(e) for e in self.events],
        }

    @staticmethod
    def _event_to_dict(e: Event) -> Dict[str, Any]:
        return {
            "channel": e.channel,
            "timestamp": float(e.timestamp),
            "value": e.value,
            "meta": e.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trace":
        events = []
        for item in d.get("events", []):
            events.append(Event(
                channel=str(item["channel"]),
                timestamp=float(item["timestamp"]),
                value=item.get("value"),
                meta=dict(item.get("meta", {})),
            ))
        return Trace(events)
