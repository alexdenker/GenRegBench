
#!/usr/bin/env bash
set -euo pipefail

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Install it with: pip install gdown"
    exit 1
fi

DATASET_DIR="dataset"
MODELS_DIR="saved_models"

mkdir -p "$DATASET_DIR"
mkdir -p "$MODELS_DIR"

echo "==> Downloading datasets..."

download_if_missing() {
    local file_id="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "  Skipping $(basename "$output") (already exists)"
    else
        echo "  Downloading $(basename "$output")..."
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
    fi
}

download_if_missing "14zyDqfEzzIvcb4F_u7DdZs1Tnwj6BGPw" "$DATASET_DIR/aapm_test.npy"
download_if_missing "1KOpm2Ho3fPsoHpEaG9x7AcW9fmtFRZQG" "$DATASET_DIR/aapm_val.npy"
download_if_missing "1-sjS7UGsmU7bVqNifCVM_T-OrRzRlAs-"  "$DATASET_DIR/afhq_test.npy"
download_if_missing "1rLGmspI7iJetSgolD0rjYrB6ctPc-R7a"  "$DATASET_DIR/afhq_val.npy"
download_if_missing "1ElTM-uG0idBkZt4q1o6D79YB8wfgT5Ht" "$DATASET_DIR/celebahq_test.npy"
download_if_missing "1edSVe3X09_d6cVudLth_K1cqKa2TE0zH" "$DATASET_DIR/celebahq_val.npy"
download_if_missing "1GahDFwQTIcbMc7deluTREa0XXA9nSizw" "$DATASET_DIR/ellipses_test.npy"
download_if_missing "1AsU6rlc2Us1eAf-SQ2I4Gk3Ehtbpy2xL" "$DATASET_DIR/ellipses_val.npy"
download_if_missing "1cHYZEnNyRzqz9-OvDe-BdDSdd0CS_qZ2" "$DATASET_DIR/ffhq_test.npy"
download_if_missing "1sjFmZBwpc1n1t1nesFcwwwDmYvvOdphr"  "$DATASET_DIR/ffhq_val.npy"
download_if_missing "19nxwboeEk487g7PI7PiU_s75rmEfFnrv" "$DATASET_DIR/walnut_test.npy"
download_if_missing "1GA4aNE7pjJVTywXYKHV7EoD3Xxfi0gnp" "$DATASET_DIR/walnut_val.npy"

echo "==> Downloading pretrained models (folder)..."
gdown --folder "https://drive.google.com/drive/folders/1ZM_JWm7dV4f7rSObgWVFw63EOLEsZREl" \
      -O "$MODELS_DIR" 

echo "==> Downloading flow models (pnpflow)..."

for dataset in celebahq ellipses aapm walnut; do
    mkdir -p "weights/${dataset}/pnpflow"
done

download_if_missing "1y4PGJRZS93Y_DMkQN_8dz5KQ-5NdacYo" "weights/celebahq/pnpflow/velocity_celebahq_final.pt"
download_if_missing "1-vZUuDMbD1oTLPoB7f6c7Pld9Ii9a4KG"  "weights/ellipses/pnpflow/velocity_ellipses_final.pt"
download_if_missing "1D7xCZ6VvftdCQthtK4lCOttYlQAbk_fV"  "weights/aapm/pnpflow/velocity_aapm_final.pt"
download_if_missing "1HFjLnHKeK7gBD56rC-_RByrcrPePmOHv"  "weights/walnut/pnpflow/velocity_walnut_final.pt"



echo "==> Done."

