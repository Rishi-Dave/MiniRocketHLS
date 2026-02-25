#!/bin/bash


# Set variables
EXECUTABLE="./minirocket_host"
MODEL_DIR="${HOME}/models/minirocket/"
USE_CASE="FruitFlies InsectSound MosquitoSound"  

make host

# Loop through each use case
for USE_CASE_NAME in $USE_CASE; do
    MODEL_FILE="${MODEL_DIR}${USE_CASE_NAME}_minirocket_model.json"
    TEST_FILE="${MODEL_DIR}${USE_CASE_NAME}_minirocket_model_test_data.json"

    echo "Running MiniRocket for use case: $USE_CASE_NAME"
    echo "Model file: $MODEL_FILE"
    echo "Test file: $TEST_FILE"

    # Execute the program
    $EXECUTABLE $1 $MODEL_FILE  $TEST_FILE > output_${USE_CASE_NAME}.txt

    echo "----------------------------------------"
done