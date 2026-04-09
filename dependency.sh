#!/usr/bin/env bash
# =============================================================================
# dependency.sh — SMoE Tokenizer Dependency Installer
# Description : Installs Rust toolchain (via rustup) and builds the tokenizer
#               extension before running requirements.txt
# Usage       : bash dependency.sh
# =============================================================================

set -euo pipefail

# ── Color & formatting helpers ────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
log_section() { echo -e "\n${BOLD}${CYAN}▶ $*${RESET}"; printf '─%.0s' {1..60}; echo; }

# ── Error handler ─────────────────────────────────────────────────────────────
on_error() {
    log_error "Script failed at line $1. Please check the output above."
    exit 1
}
trap 'on_error $LINENO' ERR

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║        SMoE Tokenizer — Dependency Setup Script          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Step 1: Update system package index ──────────────────────────────────────
log_section "Step 1/6 — Updating system package index"
sudo apt-get update -qq
log_ok "Package index updated."

# ── Step 2: Remove existing rustc/cargo (if any) ─────────────────────────────
# 若系统中已通过 apt 安装了 rustc 或 cargo，先将其卸载。
# 后续统一由 rustup 管理 Rust 工具链，避免版本冲突。
log_section "Step 2/6 — Removing system-managed Rust (if present)"
if dpkg -l rustc &>/dev/null || dpkg -l cargo &>/dev/null; then
    log_warn "Detected apt-installed rustc/cargo — removing to avoid version conflicts..."
    sudo apt-get remove -y rustc cargo
    log_ok "System rustc/cargo removed."
else
    log_info "No apt-installed rustc/cargo found, skipping removal."
fi

# ── Step 3: Install rustup via snap ──────────────────────────────────────────
# rustup 将接管 rustc 和 cargo 的版本管理，确保使用最新稳定工具链。
log_section "Step 3/6 — Installing rustup (snap)"
if command -v rustup &>/dev/null; then
    log_info "rustup already installed, skipping snap install."
else
    sudo snap install rustup --classic
    log_ok "rustup installed via snap."
fi

# ── Step 4: Configure Rust stable toolchain ──────────────────────────────────
log_section "Step 4/6 — Configuring Rust stable toolchain"
rustup default stable
rustup update
log_ok "Rust stable toolchain is up to date ($(rustc --version))."

# ── Step 5: Install Python build tools & auxiliary packages ──────────────────
log_section "Step 5/6 — Installing Python build tools and auxiliary packages"
log_info "Installing maturin (Rust↔Python build backend)..."
pip install --quiet --upgrade maturin
log_ok "maturin installed."

log_info "Installing PyYAML and huggingface-hub..."
pip install --quiet pyyaml huggingface-hub
log_ok "PyYAML and huggingface-hub installed."

# ── Step 6: Build & install tokenizer ────────────────────────────────────────
log_section "Step 6/6 — Building tokenizer (Rust extension)"
TOKENIZER_DIR="dependency/tokenizers/bindings/python"
if [[ ! -d "${TOKENIZER_DIR}" ]]; then
    log_error "Tokenizer source directory not found: ${TOKENIZER_DIR}"
    exit 1
fi
log_info "Building tokenizer in editable mode (no-build-isolation)..."
cd "${TOKENIZER_DIR}"
pip install -e . --no-build-isolation
cd - > /dev/null
log_ok "Tokenizer built and installed successfully."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}  ✔  Dependency setup complete!${RESET}"
echo -e "     You can now run:  ${BOLD}pip install -r requirements.txt${RESET}"
echo ""
