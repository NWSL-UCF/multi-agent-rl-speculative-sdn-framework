#!/bin/bash

# Define variables
BUILD_DIR="sim1_0_127"
ZIP_FILE="build.zip"
REMOTE_HANDLE="psc"
REMOTE_PATH="~/sdn/"
SSH_KEY="~/.ssh/arcc"

# Create the build directory
mkdir -p $BUILD_DIR

# Copy files to the build directory
cp runner.py psc.slurm sim.py cmd.csv $BUILD_DIR/

# Copy the Pcap folder and its contents
cp -r Pcap $BUILD_DIR/

# Create a zip archive
zip -r $ZIP_FILE $BUILD_DIR

# Transfer the zip file using scp
scp $ZIP_FILE $REMOTE_HANDLE:$REMOTE_PATH

# Clean up: remove build directory and zip file
rm -rf $BUILD_DIR $ZIP_FILE

echo "Build process completed and transferred successfully!"
