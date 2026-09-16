#!/usr/bin/env bash
#
# make_uml.sh — generate UML diagrams for the project using pyreverse.
#
# Why one-shot pyreverse instead of per-package:
#   pyreverse resolves relationships only between classes in the *same*
#   analysis pass. Running it per-package means `VQEPipeline` can't see
#   `BaseAnsatz`, `OptimizerFactory`, etc., so cross-package arrows are
#   silently dropped. One pass over every .py file fixes this.
#
# Usage:
#   ./make_uml.sh                       # all packages, all outputs
#   ./make_uml.sh core utils            # restrict to these (and their subtrees)
#   RENDER=0 ./make_uml.sh              # .puml only, no PNG/SVG
#   SHOW_PRIVATE=0 ./make_uml.sh        # hide _private members
#   MODE=classes|packages|both          # default: classes
#   CLEAN=0 ./make_uml.sh               # don't wipe OUT_DIR first
#   NO_PATH_FIX=1 ./make_uml.sh         # don't set PYTHONPATH
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# --- config ------------------------------------------------------------------
OUT_DIR="${OUT_DIR:-uml}"
MERGED_NAME="${MERGED_NAME:-all_merged}"
MODULES_NAME="${MODULES_NAME:-modules}"
PER_PACKAGE_DIR="${PER_PACKAGE_DIR:-$OUT_DIR/per_package}"
RENDER="${RENDER:-1}"
SHOW_PRIVATE="${SHOW_PRIVATE:-1}"
MODE="${MODE:-classes}"
CLEAN="${CLEAN:-1}"
NO_PATH_FIX="${NO_PATH_FIX:-0}"

mkdir -p "$OUT_DIR"

if [[ "$CLEAN" == "1" ]]; then
    find "$OUT_DIR" -maxdepth 1 -type f \
        \( -name '*.puml' -o -name '*.png' -o -name '*.svg' \) -delete
    rm -rf "$PER_PACKAGE_DIR"
fi
mkdir -p "$PER_PACKAGE_DIR"

# --- tool check --------------------------------------------------------------
if ! command -v pyreverse >/dev/null 2>&1; then
    echo "pyreverse not found. Install: pip install pylint" >&2
    exit 1
fi

# --- PYTHONPATH fix ----------------------------------------------------------
# If the project is installed with `pip install -e .`, imports resolve without
# help. If not, we need the project root on sys.path for pyreverse to resolve
# cross-package references. Setting PYTHONPATH is idempotent and harmless.
if [[ "$NO_PATH_FIX" != "1" ]]; then
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

# --- discover packages -------------------------------------------------------
discover_packages() {
    find . -type d \
        \( -name "__pycache__" -o -name ".git" -o -name ".env" \
           -o -name ".helix" -o -name "node_modules" -o -name "$OUT_DIR" \) -prune -o \
        -type d -print | while read -r dir; do
            [[ "$dir" == "." ]] && continue
            compgen -G "$dir/*.py" >/dev/null && echo "${dir#./}"
        done | sort
}

if [[ $# -gt 0 ]]; then
    PACKAGES=("$@")
else
    mapfile -t PACKAGES < <(discover_packages)
fi

# Drop packages whose ancestor is also in the list — avoids walking subtrees twice.
FILTERED=()
for p in "${PACKAGES[@]}"; do
    keep=1
    for q in "${PACKAGES[@]}"; do
        [[ "$p" == "$q" ]] && continue
        [[ "$p" == "$q/"* ]] && { keep=0; break; }
    done
    [[ $keep -eq 1 ]] && FILTERED+=("$p")
done
[[ ${#FILTERED[@]} -gt 0 ]] && PACKAGES=("${FILTERED[@]}")

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    echo "No packages found. Aborting." >&2
    exit 1
fi

echo "==> Project root  : $PROJECT_ROOT"
echo "==> Output dir    : $OUT_DIR"
echo "==> Per-package   : $PER_PACKAGE_DIR"
echo "==> Mode          : $MODE"
echo "==> Show private  : $SHOW_PRIVATE"
echo "==> Packages      : ${PACKAGES[*]}"
echo

# --- 1. sanity check ---------------------------------------------------------
echo "==> Checking imports..."
IMPORT_LIST="$(IFS=,; echo "${PACKAGES[*]//\//.}")"
if python -c "import ${IMPORT_LIST}; print('ok')" 2>/tmp/uml_import_err; then
    echo "    all packages import cleanly"
else
    echo "    WARNING: import check failed (pyreverse still parses AST):"
    sed 's/^/      /' /tmp/uml_import_err
fi
echo

# --- 2. collect all .py files ------------------------------------------------
echo "==> Collecting source files..."
PYFILES=()
for pkg in "${PACKAGES[@]}"; do
    while IFS= read -r f; do
        PYFILES+=("$f")
    done < <(find "$pkg" -name '*.py' -not -path '*/__pycache__/*' | sort)
done

if [[ ${#PYFILES[@]} -eq 0 ]]; then
    echo "No .py files found. Aborting." >&2
    exit 1
fi
echo "    ${#PYFILES[@]} files across ${#PACKAGES[@]} packages"
echo

# --- 3. one-shot pyreverse over everything -----------------------------------
echo "==> Running pyreverse (one pass over all files)..."
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PYR_OPTS=(-o puml)
[[ "$SHOW_PRIVATE" == "0" ]] && PYR_OPTS+=(--no-private)

if ! pyreverse "${PYR_OPTS[@]}" -p vqe -d "$TMPDIR" "${PYFILES[@]}" 2>/tmp/uml_pyr_err; then
    echo "pyreverse failed:" >&2
    cat /tmp/uml_pyr_err >&2
    exit 1
fi

# pyreverse writes classes_vqe.puml / packages_vqe.puml
RAW_CLASSES="$TMPDIR/classes_vqe.puml"
RAW_PACKAGES="$TMPDIR/packages_vqe.puml"

if [[ ! -s "$RAW_CLASSES" ]]; then
    echo "pyreverse produced no class diagram. Aborting." >&2
    exit 1
fi
echo "    classes  -> $RAW_CLASSES"
[[ "$MODE" == "packages" || "$MODE" == "both" ]] && \
    [[ -s "$RAW_PACKAGES" ]] && echo "    packages -> $RAW_PACKAGES"
echo

# --- 4. write merged class diagram -------------------------------------------
MERGED="$OUT_DIR/$MERGED_NAME.puml"
echo "==> Writing merged class diagram -> $MERGED"

# A user-maintained overlay lets you add arrows pyreverse can't infer
# (e.g. factory `..> : creates` links).
OVERLAY="$OUT_DIR/overlay.puml"
[[ -f "$OVERLAY" ]] || OVERLAY=""

{
    echo "@startuml VQE-Thesis"
    echo "hide empty members"
    echo "skinparam classAttributeIconSize 0"
    echo "set namespaceSeparator none"
    echo

    grep -hv \
        -e '^@startuml' \
        -e '^@enduml' \
        -e '^set namespaceSeparator' \
        -e '^hide empty members$' \
        -e '^skinparam classAttributeIconSize' \
        -e '^ *title ' \
        -e '^ *footer ' \
        "$RAW_CLASSES"

    if [[ -n "$OVERLAY" ]]; then
        echo
        echo "' --- manual overlay (from $OVERLAY) ---"
        grep -hv -e '^@startuml' -e '^@enduml' "$OVERLAY" || true
    fi

    echo "@enduml"
} > "$MERGED"

echo "    $(wc -l < "$MERGED") lines"
[[ -n "$OVERLAY" ]] && echo "    (with overlay from $OVERLAY)"

# Count arrows so we can tell at a glance whether the diagram is connected
ARROWS=$(grep -cE '(-->|<--|\*--|o--|\.\.>|<\|--)' "$MERGED" || true)
echo "    $ARROWS relationship arrows"
echo

# --- 5. per-package diagrams (supplement) ------------------------------------
if [[ "$MODE" == "both" || "$MODE" == "packages" ]]; then
    echo "==> Writing per-package diagrams -> $PER_PACKAGE_DIR"
    for pkg in "${PACKAGES[@]}"; do
        name="${pkg//\//_}"
        mapfile -t PKGFILES < <(
            find "$pkg" -name '*.py' -not -path '*/__pycache__/*' | sort
        )
        [[ ${#PKGFILES[@]} -eq 0 ]] && continue

        if pyreverse "${PYR_OPTS[@]}" -p "$name" -d "$TMPDIR" \
                "${PKGFILES[@]}" 2>/dev/null; then
            [[ "$MODE" == "both" || "$MODE" == "classes" ]] && \
                cp "$TMPDIR/classes_${name}.puml" "$PER_PACKAGE_DIR/" 2>/dev/null || true
            [[ "$MODE" == "both" || "$MODE" == "packages" ]] && \
                cp "$TMPDIR/packages_${name}.puml" "$PER_PACKAGE_DIR/" 2>/dev/null || true
            echo "    $pkg"
        fi
    done
    echo
fi

# --- 6. module-level component diagram ---------------------------------------
MODULES_PUML="$OUT_DIR/$MODULES_NAME.puml"
echo "==> Building module dependency diagram -> $MODULES_PUML"

{
    echo "@startuml VQE-Modules"
    echo "!theme plain"
    echo "skinparam componentStyle rectangle"
    echo "skinparam linetype ortho"
    echo "hide stereotype"
    echo
    echo "title VQE-Thesis — module dependencies"
    echo

    for p in "${PACKAGES[@]}"; do
        npy=$(find "$p" -name '*.py' -not -path '*/__pycache__/*' | wc -l)
        ncls=$(grep -rhE '^\s*class ' "$p" --include='*.py' 2>/dev/null | wc -l)
        echo "component \"$p\\n(${npy} files, ${ncls} classes)\" as C_${p//\//_}"
    done
    echo

    for src in "${PACKAGES[@]}"; do
        for dst in "${PACKAGES[@]}"; do
            [[ "$src" == "$dst" ]] && continue
            if grep -rqE "^\s*(from\s+${dst//\//.}(\.|\s)|import\s+${dst//\//.}(\.|\s|$))" \
                    "$src" --include='*.py' 2>/dev/null; then
                echo "C_${src//\//_} ..> C_${dst//\//_} : <<import>>"
            fi
        done
    done
    echo "@enduml"
} > "$MODULES_PUML"

MOD_ARROWS=$(grep -cE '\.\.>' "$MODULES_PUML" || true)
echo "    $(wc -l < "$MODULES_PUML") lines, $MOD_ARROWS dependency arrows"
echo

# --- 7. render ---------------------------------------------------------------
if [[ "$RENDER" != "1" ]]; then
    echo "==> RENDER=0, skipping plantuml."
    echo
    echo "Done. Outputs in $OUT_DIR/:"
    find "$OUT_DIR" -maxdepth 1 -type f | sort
    exit 0
fi

echo "==> Rendering with plantuml..."

RENDER_ALL=("$MERGED" "$MODULES_PUML")

if command -v plantuml >/dev/null 2>&1; then
    plantuml -tpng "${RENDER_ALL[@]}" >/dev/null
    plantuml -tsvg "${RENDER_ALL[@]}" >/dev/null
elif [[ -f plantuml.jar ]]; then
    java -jar plantuml.jar -tpng "${RENDER_ALL[@]}" >/dev/null
    java -jar plantuml.jar -tsvg "${RENDER_ALL[@]}" >/dev/null
else
    echo "    plantuml not found. Install it or drop plantuml.jar here."
    echo "    .puml files are ready in $OUT_DIR/."
    exit 0
fi

echo
echo "==> Done."
echo "    $OUT_DIR/$MERGED_NAME.puml       ($ARROWS arrows)"
echo "    $OUT_DIR/$MODULES_NAME.puml      ($MOD_ARROWS arrows)"
echo "    $OUT_DIR/*.png / *.svg"
