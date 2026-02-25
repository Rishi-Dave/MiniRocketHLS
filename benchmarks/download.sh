#!/bin/bash

DATASETS="InsectSound FruitFlies MosquitoSound"


for DATASET in $DATASETS; do

    URL="https://www.timeseriesclassification.com/aeon-toolkit/$DATASET.zip"
    ZIP_FILE="$DATASET.zip"
    TARGET_DIR="$DATASET"

    if [ ! -f "$ZIP_FILE" ]; then
        echo "Downloading $ZIP_FILE..."
        wget "$URL"
    else
        echo "$ZIP_FILE already exists. Skipping download."
    fi

    # Create target directory if it doesn't exist
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Creating directory $TARGET_DIR..."
        mkdir -p "$TARGET_DIR"
    else
        echo "Directory $TARGET_DIR already exists."
    fi

    # Unzip into target directory
    echo "Unzipping into $TARGET_DIR..."
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

done

echo "Done."