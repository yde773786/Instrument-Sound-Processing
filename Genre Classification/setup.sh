#!/bin/bash

# Run: ./setup.sh [0|1] [0|1]
# First argument is: setup (1), unsetup (0)
# Second argument is: installed (1), not installed (0)
UNMOUNT=$1
INSTALLED=$2

# For consistency, this script can be run to mount a google drive containing the GTZAN dataset.
# This sets up the environment as required for local development.
# Not required for Google Collaboratory (as already mounted).
# Assumes GTZAN is mounted in Google Drive

if [ "$INSTALLED" -eq 0 ]; then
  sudo add-apt-repository ppa:alessandro-strada/ppa
  sudo apt-get update
  sudo apt-get install google-drive-ocamlfuse
fi

# mount at home directory
mkdir ~/google_drive
google-drive-ocamlfuse ~/google_drive

if [ "$UNMOUNT" -eq 0 ]; then
    fusermount -u ~/google_drive
else
    # Create symlink to mount-point
    sudo mkdir -p /content/drive/
    sudo ln -s ~/google_drive /content/drive/MyDrive
fi
