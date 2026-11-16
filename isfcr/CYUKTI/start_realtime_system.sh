#!/bin/bash
#
# start_realtime_system.sh
# ========================
# Quick-start script for CYUKTI real-time SOC system
#
# Usage: ./start_realtime_system.sh [component]
#   component: backend|listener|frontend|all|test
#
# Requirements:
# - Neo4j running (docker start neo4j-soc)
# - Wazuh manager active
# - soc_env Python environment
# - Node.js/npm for frontend

set -e  # Exit on error

CAPSTONE_ROOT="$HOME/capstone"
BACKEND_DIR="$CAPSTONE_ROOT/isfcr/CYUKTI/backend"
FRONTEND_DIR="$CAPSTONE_ROOT/isfcr/CYUKTI/frontend"
VENV="$CAPSTONE_ROOT/soc_env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_neo4j() {
    log_info "Checking Neo4j connection..."
    if docker ps | grep -q neo4j-soc; then
        log_success "Neo4j container running"
        return 0
    else
        log_error "Neo4j not running"
        log_info "Start with: docker start neo4j-soc"
        return 1
    fi
}

check_wazuh() {
    log_info "Checking Wazuh manager..."
    if systemctl is-active --quiet wazuh-manager 2>/dev/null || sudo systemctl is-active --quiet wazuh-manager 2>/dev/null; then
        log_success "Wazuh manager active"
        return 0
    else
        log_warning "Wazuh manager status unknown (may require sudo)"
        return 0  # Don't fail on this
    fi
}

check_venv() {
    log_info "Checking Python virtual environment..."
    if [ -d "$VENV" ]; then
        log_success "Virtual environment found: $VENV"
        return 0
    else
        log_error "Virtual environment not found: $VENV"
        return 1
    fi
}

start_backend() {
    log_info "Starting CYUKTI backend with WebSocket support..."
    
    cd "$BACKEND_DIR"
    source "$VENV/bin/activate"
    
    log_info "Backend will run on http://0.0.0.0:5002"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    python dashboard_api.py
}

start_listener() {
    log_info "Starting Wazuh listener..."
    
    cd "$BACKEND_DIR"
    source "$VENV/bin/activate"
    
    log_info "Monitoring: /var/ossec/logs/alerts/alerts.json"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    python listener/wazuh_listener.py
}

start_frontend() {
    log_info "Starting frontend dashboard..."
    
    cd "$FRONTEND_DIR"
    
    if [ ! -d "node_modules" ]; then
        log_warning "node_modules not found, running npm install..."
        npm install
    fi
    
    log_info "Dashboard will run on http://localhost:5173"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    npm run dev
}

run_tests() {
    log_info "Running integration tests..."
    
    cd "$BACKEND_DIR"
    source "$VENV/bin/activate"
    
    python test_realtime_pipeline.py
}

show_usage() {
    cat <<EOF
CYUKTI Real-Time System Startup

Usage: $0 [component]

Components:
  backend   - Start Flask backend with WebSocket support (port 5002)
  listener  - Start Wazuh alert listener
  frontend  - Start Vite frontend dashboard (port 5173)
  all       - Show instructions for starting all components
  test      - Run integration tests
  help      - Show this help message

Examples:
  $0 backend          # Start backend only
  $0 listener         # Start listener only
  $0 frontend         # Start frontend only
  $0 test             # Run tests
  $0 all              # Show full startup guide

Quick Start (requires 3 terminals):
  Terminal 1: $0 backend
  Terminal 2: $0 listener
  Terminal 3: $0 frontend
EOF
}

start_all_guide() {
    cat <<EOF
${BLUE}========================================
CYUKTI REAL-TIME SYSTEM - FULL STARTUP
========================================${NC}

This system requires 3 separate terminal windows.

${GREEN}Terminal 1 - Backend:${NC}
    cd ~/capstone/isfcr/CYUKTI
    ./start_realtime_system.sh backend

${GREEN}Terminal 2 - Listener:${NC}
    cd ~/capstone/isfcr/CYUKTI
    ./start_realtime_system.sh listener

${GREEN}Terminal 3 - Frontend:${NC}
    cd ~/capstone/isfcr/CYUKTI
    ./start_realtime_system.sh frontend

${YELLOW}Verification:${NC}
1. Open browser to http://localhost:5173
2. Check browser console (F12): "WebSocket connected to CYUKTI backend"
3. Dashboard shows green WiFi icon in Live SOC Feed
4. Generate test alert: 
   echo '{"timestamp":"2024-11-15T12:00:00Z","rule":{"id":"60204","description":"Test SSH Attack","level":10},"data":{"srcip":"203.0.113.50","dstip":"192.168.56.105"}}' | sudo tee -a /var/ossec/logs/alerts/alerts.json
5. Event appears in dashboard within 1 second

${YELLOW}Testing:${NC}
    ./start_realtime_system.sh test

${YELLOW}Documentation:${NC}
    REALTIME_DEPLOYMENT.md - Complete deployment guide
    REALTIME_IMPLEMENTATION_REPORT.md - Technical details

${BLUE}========================================${NC}
EOF
}

# Main script logic
main() {
    COMPONENT="${1:-help}"
    
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════╗"
    echo "║   CYUKTI Real-Time System Startup   ║"
    echo "╔══════════════════════════════════════╗"
    echo -e "${NC}"
    echo ""
    
    case "$COMPONENT" in
        backend)
            check_neo4j || exit 1
            check_wazuh
            check_venv || exit 1
            start_backend
            ;;
        listener)
            check_venv || exit 1
            start_listener
            ;;
        frontend)
            start_frontend
            ;;
        all)
            check_neo4j || exit 1
            check_wazuh
            check_venv || exit 1
            start_all_guide
            ;;
        test)
            check_neo4j || exit 1
            check_venv || exit 1
            run_tests
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown component: $COMPONENT"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
