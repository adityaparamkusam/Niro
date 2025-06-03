set -euo pipefail

BASE_DIR="$HOME/Niro/Data/wiki_only"
mkdir -p "$BASE_DIR" && cd "$BASE_DIR"

log(){ echo "[$(date '+%F %T')] $*"; }

# check ~15 GB free
free_gb=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4);print $4}')
(( free_gb < 20 )) && { echo "Need ~15 GB, only ${free_gb} GB free"; exit 1; }

# prerequisites
for t in wget python3; do command -v $t >/dev/null || { echo "$t required"; exit 1; }; done
python3 - <<'PY' 2>/dev/null || pip install --user wikiextractor
import wikiextractor, sys; sys.exit(0)
PY

# download
log "Fetching enwiki dump (~15 GB bz2)…"
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
     -O enwiki.xml.bz2 --progress=bar:force:noscroll

# extract
log "Running WikiExtractor…"
python3 -m wikiextractor.WikiExtractor enwiki.xml.bz2 \
        --processes 2 --bytes 100M --output extracted

log "✅ Wikipedia ready in $BASE_DIR/extracted"
