"""Shared library for the Mosaic theologian pipeline.

Modules here run unchanged on the Windows build machine and the Mac query
machine. Anything CUDA-, Whisper-, or crawler-specific lives in workers/.
"""
