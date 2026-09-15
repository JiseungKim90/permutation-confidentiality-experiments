# Runtime inputs

Run `python3 scripts/download_inputs.py` from the parent `tdsc/` directory.
The downloaded checkpoint and archive are ignored by Git. The downloader
verifies their complete SHA-256 digests before promoting temporary downloads
to their final filenames. The runtime loader reads the named CIFAR batch
directly from the authenticated archive and does not extract it.
