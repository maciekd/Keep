#!/usr/bin/env python3
"""Takes the recovery partition out of a stopped VM's disk image, so that the system's container
can grow into the space `tart set --disk-size` added behind it.

A macOS guest's disk is: the boot container, the system container, the recovery container — and
then the new free space, which the system cannot reach across the recovery partition. macOS
refuses to remove that partition while System Integrity Protection is on, in the guest and on an
attached image alike, so this edits the image's partition table itself: the recovery entry is
cleared in the primary and the backup table and their checksums are rewritten. The guest boots
without it (it only loses its recovery mode) and `diskutil apfs resizeContainer disk0s2 0` then
takes the whole disk. Only ever for a throwaway test VM.

    free_disk.py <disk.img>
"""
import struct
import sys
import zlib

SECTOR = 512
RECOVERY = "52637672-7900-11AA-AA11-00306543ECAC"      # Apple_APFS_Recovery


def guid(raw: bytes) -> str:
    a, b, c = struct.unpack("<IHH", raw[:8])
    return f"{a:08X}-{b:04X}-{c:04X}-{raw[8:10].hex().upper()}-{raw[10:16].hex().upper()}"


def rewrite(image, header_lba: int) -> bool:
    """Clears the recovery entry of the table whose header is at *header_lba*; whether it had one."""
    image.seek(header_lba * SECTOR)
    header = bytearray(image.read(SECTOR))
    if header[:8] != b"EFI PART":
        sys.exit(f"no partition table header at sector {header_lba}")
    size = struct.unpack_from("<I", header, 12)[0]
    entries_lba, count, entry_size = struct.unpack_from("<QII", header, 72)
    image.seek(entries_lba * SECTOR)
    entries = bytearray(image.read(count * entry_size))
    found = False
    for i in range(count):
        entry = entries[i * entry_size:(i + 1) * entry_size]
        if guid(entry[:16]) == RECOVERY:
            entries[i * entry_size:(i + 1) * entry_size] = bytes(entry_size)
            found = True
    if not found:
        return False
    struct.pack_into("<I", header, 88, zlib.crc32(entries) & 0xFFFFFFFF)
    struct.pack_into("<I", header, 16, 0)
    struct.pack_into("<I", header, 16, zlib.crc32(header[:size]) & 0xFFFFFFFF)
    image.seek(entries_lba * SECTOR)
    image.write(entries)
    image.seek(header_lba * SECTOR)
    image.write(header)
    return True


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], "r+b") as image:
        image.seek(SECTOR)
        backup_lba = struct.unpack_from("<Q", image.read(SECTOR), 32)[0]
        primary = rewrite(image, 1)
        backup = rewrite(image, backup_lba)
    print("recovery partition removed" if primary or backup else "no recovery partition: nothing to do")


if __name__ == "__main__":
    main()
