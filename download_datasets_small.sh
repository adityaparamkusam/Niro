#!/bin/bash

set -e

# Base directory setup
BASE_DIR="$HOME/Niro/Data"
echo "Creating directory structure in: $BASE_DIR"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# Create relevant subdirectories
mkdir -p {wikipedia,programming_docs,customer_service,encyclopedic,fact_databases}

# Function to log progress
log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BASE_DIR/download.log"
}

# Function to check available disk space (cross-platform)
check_disk_space() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_gb=$(df -g . | awk 'NR==2 {print $4}')
    else
        available_gb=$(df -BG . | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
    fi

    if [ "$available_gb" -lt 100 ]; then
        echo "Warning: Only ${available_gb}GB available. This download requires ~50 GB."
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Enhanced download function with retries and progress bar
download_with_retry() {
    local url="$1"
    local output="$2"
    local retries=3

    for ((i=1; i<=retries; i++)); do
        if wget -c --timeout=30 --tries=3 --progress=bar:force:noscroll "$url" -O "$output"; then
            return 0
        fi
        log_progress "Download attempt $i failed for $url"
        sleep 5
    done
    return 1
}

# Check prerequisites
check_prerequisites() {
    log_progress "Checking prerequisites..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "Homebrew required. Install from: https://brew.sh"
            exit 1
        fi
        PKG_INSTALL="brew install"
    elif command -v apt-get >/dev/null 2>&1; then
        PKG_INSTALL="sudo apt-get install -y"
    elif command -v yum >/dev/null 2>&1; then
        PKG_INSTALL="sudo yum install -y"
    else
        echo "Unsupported package manager."
        exit 1
    fi

    for tool in wget python3 git; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "$tool required. Install with: $PKG_INSTALL $tool"
            exit 1
        fi
    done

    if ! command -v 7z >/dev/null 2>&1 && ! command -v 7za >/dev/null 2>&1; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "7zip required. Install with: brew install p7zip"
        else
            echo "7zip required. Install with: $PKG_INSTALL p7zip-full"
        fi
        exit 1
    fi

    for package in requests beautifulsoup4; do
        python3 -c "import $package" 2>/dev/null || {
            echo "Installing $package...";
            python3 -m pip install --user $package;
        }
    done

    if command -v kaggle >/dev/null 2>&1; then
        if [ -f "$BASE_DIR/kaggle.json" ]; then
            export KAGGLE_CONFIG_DIR="$BASE_DIR"
            log_progress "Found kaggle.json in project directory"
        elif [ -f ~/.kaggle/kaggle.json ]; then
            log_progress "Found kaggle.json in standard location"
        else
            log_progress "Kaggle API not configured"
        fi
    else
        log_progress "Kaggle CLI not found"
    fi
}

# Wikipedia (~15 GB)
download_wikipedia() {
    log_progress "Starting Wikipedia download..."
    cd "$BASE_DIR/wikipedia"

    log_progress "Downloading English Wikipedia dump..."
    if ! download_with_retry "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" "enwiki-latest-pages-articles.xml.bz2"; then
        download_with_retry "https://dumps.wikimedia.org/enwiki/20240101/enwiki-20240101-pages-articles.xml.bz2" "enwiki-latest-pages-articles.xml.bz2" || true
    fi

    python3 -c "import wikiextractor" 2>/dev/null || {
        log_progress "Installing WikiExtractor..."
        python3 -m pip install --user wikiextractor
    }

    if [ -f "enwiki-latest-pages-articles.xml.bz2" ]; then
        log_progress "Extracting Wikipedia text..."
        python3 -m wikiextractor.WikiExtractor --processes 2 --output wikipedia_extracted --bytes 100M enwiki-latest-pages-articles.xml.bz2 || {
            log_progress "WikiExtractor failed"
        }
    fi

    log_progress "Wikipedia download completed"
}

# Programming Documentation (~2 GB)
download_programming_docs() {
    log_progress "Starting Programming Documentation download..."
    cd "$BASE_DIR/programming_docs"

    log_progress "Downloading Python documentation..."
    PYTHON_VERSION="3.12.0"
    if ! download_with_retry "https://docs.python.org/3/archives/python-${PYTHON_VERSION}-docs-text.tar.bz2" "python-docs.tar.bz2"; then
        download_with_retry "https://docs.python.org/3/archives/python-3.11.0-docs-text.tar.bz2" "python-docs.tar.bz2" || true
    fi
    [ -f "python-docs.tar.bz2" ] && tar -xjf python-docs.tar.bz2 2>/dev/null || true

    log_progress "Cloning DevDocs..."
    git clone --depth 1 --single-branch https://github.com/freeCodeCamp/devdocs.git devdocs 2>/dev/null || true

    log_progress "Downloading MDN content samples..."
    mkdir -p mdn_samples
    download_with_retry "https://raw.githubusercontent.com/mdn/content/main/README.md" "mdn_samples/README.md" || true
    download_with_retry "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/index.md" "mdn_samples/javascript.md" || true

    log_progress "Programming documentation download completed"
}

# Customer Service (~5 GB)
download_customer_service() {
    log_progress "Starting Customer Service data download..."
    cd "$BASE_DIR/customer_service"

    if command -v kaggle >/dev/null 2>&1 && [ -f "$HOME/.kaggle/kaggle.json" -o -f "$BASE_DIR/kaggle.json" ]; then
        [ -f "$BASE_DIR/kaggle.json" ] && export KAGGLE_CONFIG_DIR="$BASE_DIR"
        kaggle datasets download -d thoughtvector/customer-support-on-twitter --unzip 2>/dev/null || true
    fi

    log_progress "Creating sample customer service data..."
    cat > sample_customer_service.csv << 'EOF'
customer_message,agent_response,category
"I can't log into my account","Please try resetting your password using the forgot password link",login_issue
"My order hasn't arrived","Let me check the tracking information for your order",shipping_inquiry
"I want to return this item","I can help you with that. What's the reason for the return?",return_request
EOF

    log_progress "Customer service data download completed"
}

# Encyclopedic Content (~10 GB)
download_encyclopedic() {
    log_progress "Starting Encyclopedic Content download..."
    cd "$BASE_DIR/encyclopedic"

    log_progress "Downloading Project Gutenberg catalog..."
    download_with_retry "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv" "pg_catalog.csv" || true

    log_progress "Downloading popular Project Gutenberg books..."
    popular_books=(74 1342 11 84 1080 2701 345 1184 46 76)

    for book_id in "${popular_books[@]}"; do
        download_with_retry "https://www.gutenberg.org/files/${book_id}/${book_id}-0.txt" "book_${book_id}.txt" || \
        download_with_retry "https://www.gutenberg.org/files/${book_id}/${book_id}.txt" "book_${book_id}.txt" || true
    done

    log_progress "Downloading Simple English Wikipedia..."
    download_with_retry "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2" "simplewiki.xml.bz2" || true

    log_progress "Encyclopedic content download completed"
}

# Fact Databases (~3 GB)
download_fact_databases() {
    log_progress "Starting Fact Databases download..."
    cd "$BASE_DIR/fact_databases"

    log_progress "Downloading YAGO sample..."
    download_with_retry "https://yago-knowledge.org/data/yago4.5/en/yago-wd-simple-types.nt.gz" "yago-types.nt.gz" || true

    if [ ! -f "../knowledge_bases/conceptnet.csv.gz" ]; then
        log_progress "Downloading ConceptNet assertions..."
        download_with_retry "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz" "conceptnet.csv.gz" || true
    fi

    log_progress "Downloading WordNet..."
    download_with_retry "https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz" "wordnet.tar.gz" || true
    [ -f "wordnet.tar.gz" ] && tar -xzf wordnet.tar.gz 2>/dev/null || true

    log_progress "Creating sample fact database..."
    cat > sample_facts.json << 'EOF'
{
  "facts": [
    {"subject": "Earth", "predicate": "is_a", "object": "planet"},
    {"subject": "Python", "predicate": "is_a", "object": "programming_language"},
    {"subject": "Shakespeare", "predicate": "wrote", "object": "Hamlet"}
  ]
}
EOF

    log_progress "Fact databases download completed"
}

# Main execution
main() {
    log_progress "Starting dataset download (<20GB)..."
    log_progress "Target directory: $BASE_DIR"

    check_disk_space
    check_prerequisites

    exec > >(tee -a "$BASE_DIR/download.log") 2>&1

    local total_steps=5
    local current_step=0

    for download_func in \
        "download_wikipedia" \
        "download_programming_docs" \
        "download_customer_service" \
        "download_encyclopedic" \
        "download_fact_databases"; do

        current_step=$((current_step + 1))
        log_progress "=== Step $current_step/$total_steps: Running $download_func ==="

        if $download_func; then
            log_progress "✓ $download_func completed successfully"
        else
            log_progress "⚠ $download_func completed with some errors"
        fi

        log_progress "Progress: $current_step/$total_steps steps completed"
    done

    log_progress "All downloads completed!"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        total_size=$(du -sh "$BASE_DIR" | cut -f1)
    else
        total_size=$(du -sh "$BASE_DIR" | cut -f1)
    fi

    log_progress "Total directory size: $total_size"

    {
        echo "Dataset Download Summary (<20GB)"
        echo "=============================="
        echo "Download completed: $(date)"
        echo "Base directory: $BASE_DIR"
        echo "Total size: $total_size"
        echo "Free space remaining: $(df -h . | awk 'NR==2 {print $4}')"
        echo ""
        echo "Directory sizes:"
        du -sh "$BASE_DIR"/* 2>/dev/null | sort -hr
    } > "$BASE_DIR/download_summary.txt"

    log_progress "Summary report created: $BASE_DIR/download_summary.txt"
}

if ! main "$@"; then
    log_progress "Script encountered errors. Check $BASE_DIR/download.log."
    exit 1
fi