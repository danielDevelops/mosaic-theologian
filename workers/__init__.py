"""Batch workers invoked by Mosaic-NightJob.ps1.

Each worker does one pipeline stage, honours a wall-clock deadline between
items, and records progress only after the artifact it describes is on disk.
Workers are safe to kill and safe to re-run.
"""
