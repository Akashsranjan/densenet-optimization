OUTPUT_DIR="./results"
GPU_ENABLED=false


while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --gpu-enabled) GPU_ENABLED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting MLOps Engineer Benchmarking Assignment..."

mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs/tensorboard


echo "Building Docker image..."
docker-compose build


echo "Running benchmarking containers..."
if [ "$GPU_ENABLED" = true ]; then
    echo "Running with GPU support enabled."
    docker-compose up --build --abort-on-container-exit --remove-orphans
else
    echo "Running in CPU-only mode."
    docker-compose up --build --abort-on-container-exit --remove-orphans --no-deps benchmark-app
fi

echo "Benchmarking completed. Results are in $OUTPUT_DIR and TensorBoard logs are in ./logs/tensorboard."
echo "To view TensorBoard, run: docker-compose up tensorboard"

docker-compose down
echo "Done."