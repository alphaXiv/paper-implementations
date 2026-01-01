#!/bin/bash
# DeepCode Speedrun Setup Script
# Follows the official DeepCode installation recipe using UV

set -e  # Exit on error

# Color codes for terminal output
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
BLUE='\033[94m'
BOLD='\033[1m'
ENDC='\033[0m'

# Configuration files
CONFIG_FILE="mcp_agent.config.yaml"
SECRETS_FILE="mcp_agent.secrets.yaml"

# Padding constant
PADDING="  "

# Override read to automatically add padding to prompts
read() {
    if [ "$1" = "-p" ]; then
        shift
        local prompt="$1"
        shift
        # Check if prompt already starts with padding
        if [[ "$prompt" =~ ^${PADDING} ]]; then
            command read -p "$prompt" "$@"
        else
            command read -p "${PADDING}${prompt}" "$@"
        fi
    else
        command read "$@"
    fi
}

print_header() {
    clear
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         🚀 DeepCode Speedrun Setup 🚀                    ║"
    echo "║     Automated Configuration & Installation Wizard        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${ENDC}\n"
}

print_step() {
    local step_num=$1
    local message=$2
    echo -e "\n${PADDING}${CYAN}${BOLD}Step ${step_num}:${ENDC} ${message}"
}

print_success() {
    echo -e "${PADDING}${GREEN}✅ $1${ENDC}"
}

print_warning() {
    echo -e "${PADDING}${YELLOW}⚠️  $1${ENDC}"
}

print_error() {
    echo -e "${PADDING}${RED}❌ $1${ENDC}"
}

print_info() {
    echo -e "${PADDING}${BLUE}ℹ️  $1${ENDC}"
}

install_uv() {
    local step_num=$1
    print_step "$step_num" "Installing UV package manager..."
    
    if command -v uv &> /dev/null; then
        print_success "UV is already installed"
        return 0
    fi
    
    print_info "Installing UV..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        print_success "UV installed successfully"
        
        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Try to source the shell profile to get UV in PATH
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
        if [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc" 2>/dev/null || true
        fi
        
        # Check if uv is now available
        if ! command -v uv &> /dev/null; then
            print_warning "UV installed but not in PATH. You may need to restart your terminal."
            print_info "Or run: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        fi
        return 0
    else
        print_error "Failed to install UV"
        return 1
    fi
}

setup_venv() {
    local step_num=$1
    print_step "$step_num" "Setting up Python virtual environment with UV..."
    
    # Ensure UV is in PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        print_error "UV not found. Please install it first or restart your terminal."
        exit 1
    fi
    
    # Check Python 3.13 availability
    if command -v python3.13 &> /dev/null; then
        PYTHON_VERSION="3.13"
    elif python3 -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        print_warning "Python 3.13 not found, using Python $PYTHON_VERSION"
    else
        print_error "Python 3.13+ required. Please install Python 3.13 or later."
        exit 1
    fi
    
    if [ -d ".venv" ]; then
        print_warning ".venv already exists"
        read -p "Recreate virtual environment? (y/N): " recreate_venv
        if [[ "$recreate_venv" =~ ^[Yy]$ ]]; then
            rm -rf .venv
            uv venv --python="$PYTHON_VERSION"
            print_success "Virtual environment recreated"
        else
            print_info "Using existing virtual environment"
        fi
    else
        uv venv --python="$PYTHON_VERSION"
        print_success "Virtual environment created"
    fi
    
    # Activate venv (Linux/macOS)
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    # Activate venv (Windows)
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi
    
    # Verify activation worked
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "Virtual environment may not be activated properly"
        print_info "Continuing anyway..."
    else
        print_info "Virtual environment active: $VIRTUAL_ENV"
    fi
}

install_dependencies() {
    local step_num=$1
    print_step "$step_num" "Installing dependencies with UV..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Ensure PyYAML is in requirements
    if ! grep -qi "pyyaml" requirements.txt; then
        print_warning "PyYAML not found in requirements.txt, adding it..."
        echo "pyyaml>=6.0" >> requirements.txt
    fi
    
    print_info "Installing dependencies (this may take a moment)..."
    
    # Ensure UV is in PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if uv pip install -r requirements.txt; then
        print_success "Dependencies installed successfully"
        return 0
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# YAML editing functions using Python
set_yaml_value() {
    local file=$1
    local key=$2
    local value=$3
    
    python3 << EOF
import yaml
import sys

try:
    with open("$file", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    keys = "$key".split('.')
    current = config
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = "$value"
    
    with open("$file", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

install_windows_mcp_servers() {
    # Detect Windows (Git Bash, WSL, or native Windows)
    if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" && "$OSTYPE" != "cygwin" ]]; then
        # Not Windows, skip this step
        return 0
    fi
    
    echo ""
    print_info "Windows detected. Installing MCP servers globally..."
    echo ""
    echo "  The following commands will be run:"
    echo "    ${CYAN}npm i -g @modelcontextprotocol/server-brave-search${ENDC}"
    echo "    ${CYAN}npm i -g @modelcontextprotocol/server-filesystem${ENDC}"
    echo ""
    read -p "Install MCP servers now? (Y/n): " install_servers
    if [[ "$install_servers" =~ ^[Nn]$ ]]; then
        print_warning "Skipping MCP server installation."
        print_info "You can install them manually later."
        return 0
    fi
    
    echo ""
    print_info "Installing @modelcontextprotocol/server-brave-search..."
    if npm i -g @modelcontextprotocol/server-brave-search; then
        print_success "Installed @modelcontextprotocol/server-brave-search"
    else
        print_error "Failed to install @modelcontextprotocol/server-brave-search"
        return 1
    fi
    
    echo ""
    print_info "Installing @modelcontextprotocol/server-filesystem..."
    if npm i -g @modelcontextprotocol/server-filesystem; then
        print_success "Installed @modelcontextprotocol/server-filesystem"
    else
        print_error "Failed to install @modelcontextprotocol/server-filesystem"
        return 1
    fi
    
    return 0
}

configure_windows_mcp_servers() {
    # Detect Windows (Git Bash, WSL, or native Windows)
    if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" && "$OSTYPE" != "cygwin" ]]; then
        # Not Windows, skip this step
        return 0
    fi
    
    local step_num=$1
    print_step "$step_num" "Configuring Windows MCP Servers"
    
    echo ""
    print_info "Windows detected. MCP servers need to be configured with absolute paths."
    echo ""
    read -p "Configure Windows MCP servers? (Y/n): " configure_windows
    if [[ "$configure_windows" =~ ^[Nn]$ ]]; then
        print_info "Skipping Windows MCP server configuration"
        return 0
    fi
    
    echo ""
    print_info "Finding your global node_modules path..."
    echo ""
    print_info "Running: npm -g root"
    npm_root=$(npm -g root 2>/dev/null)
    
    if [ -z "$npm_root" ]; then
        print_error "Failed to get npm global root path."
        print_info "You can configure this manually later by editing $CONFIG_FILE"
        return 1
    fi
    
    print_success "Found npm global root: $npm_root"
    
    # Build Windows paths with forward slashes (YAML format)
    local brave_path="${npm_root}/@modelcontextprotocol/server-brave-search/dist/index.js"
    local filesystem_path="${npm_root}/@modelcontextprotocol/server-filesystem/dist/index.js"
    
    # Convert to Windows path format if needed (Git Bash/Cygwin)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Git Bash/Cygwin - convert Unix path to Windows path
        brave_path=$(cygpath -w "$brave_path" 2>/dev/null || echo "$brave_path")
        filesystem_path=$(cygpath -w "$filesystem_path" 2>/dev/null || echo "$filesystem_path")
    fi
    
    # Convert backslashes to forward slashes for YAML (Windows paths use forward slashes in YAML)
    brave_path=$(echo "$brave_path" | sed 's|\\|/|g')
    filesystem_path=$(echo "$filesystem_path" | sed 's|\\|/|g')
    
    echo ""
    print_info "Updating $CONFIG_FILE with Windows paths..."
    print_info "Brave path: $brave_path"
    print_info "Filesystem path: $filesystem_path"
    
    # Update config using Python YAML
    python3 << EOF
import yaml
import sys

try:
    with open("$CONFIG_FILE", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Update brave server for Windows
    if 'mcp' in config and 'servers' in config['mcp'] and 'brave' in config['mcp']['servers']:
        config['mcp']['servers']['brave']['command'] = 'node'
        config['mcp']['servers']['brave']['args'] = ["$brave_path"]
    
    # Update filesystem server for Windows
    if 'mcp' in config and 'servers' in config['mcp'] and 'filesystem' in config['mcp']['servers']:
        config['mcp']['servers']['filesystem']['command'] = 'node'
        config['mcp']['servers']['filesystem']['args'] = ["$filesystem_path", "."]
    
    with open("$CONFIG_FILE", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print("Updated Windows MCP server configuration")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Updated Windows MCP server configuration"
    else
        print_error "Failed to update Windows MCP server configuration"
        print_info "You can configure this manually by editing $CONFIG_FILE"
        return 1
    fi
    
    return 0
}

configure_search_server() {
    local step_num=$1
    print_step "$step_num" "Configuring Search Server (Optional)"
    
    echo ""
    echo "  Search servers enable web search functionality for finding similar repositories"
    echo "  and code examples. This is optional - you can skip this step."
    echo ""
    print_warning "Without a search API key, the following features will be disabled:"
    echo "    • Searching for similar repositories"
    echo "    • Finding code examples from the web"
    echo "    • Web-based code reference lookup"
    echo ""
    echo "  Available search servers:"
    echo "    [1] brave"
    echo "    [2] bocha-mcp"
    echo "    [3] Skip / None"
    echo ""
    
    read -p "Select search server (1-3, default: 3): " SELECTION
    SELECTION=${SELECTION:-3}
    
    if [[ ! "$SELECTION" =~ ^[0-9]+$ ]] || [ "$SELECTION" -lt 1 ] || [ "$SELECTION" -gt 3 ]; then
        print_info "Skipping search server configuration"
        return 0
    fi
    
    if [ "$SELECTION" -eq 3 ]; then
        print_info "Skipping search server configuration"
        echo ""
        print_info "You can configure this later by editing $CONFIG_FILE"
        return 0
    fi
    
    local server=""
    case $SELECTION in
        1) server="brave" ;;
        2) server="bocha-mcp" ;;
        *) return 0 ;;
    esac
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "$CONFIG_FILE not found!"
        return 1
    fi
    
    set_yaml_value "$CONFIG_FILE" "default_search_server" "$server"
    print_success "Set default_search_server to: $server"
    
    echo ""
    # Convert server name to uppercase for display (portable method)
    local server_upper=$(echo "$server" | tr '[:lower:]' '[:upper:]')
    read -p "Enter $server_upper API key (optional, press Enter to skip): " api_key
    
    if [ -n "$api_key" ]; then
        if [ "$server" = "brave" ]; then
            set_yaml_value "$CONFIG_FILE" "mcp.servers.brave.env.BRAVE_API_KEY" "$api_key"
            print_success "Set BRAVE_API_KEY"
        elif [ "$server" = "bocha-mcp" ]; then
            set_yaml_value "$CONFIG_FILE" "mcp.servers.bocha-mcp.env.BOCHA_API_KEY" "$api_key"
            print_success "Set BOCHA_API_KEY"
        fi
    else
        print_warning "No API key provided."
        echo ""
        print_warning "Without a search API key, the following features will be disabled:"
        echo "    • Searching for similar repositories"
        echo "    • Finding code examples from the web"
        echo "    • Web-based code reference lookup"
        echo ""
        print_info "You can add the API key later by editing $CONFIG_FILE"
    fi
    
    return 0
}

configure_api_keys() {
    local step_num=$1
    print_step "$step_num" "Configuring API Keys"
    
    print_info "Configure at least one API key. The first provider you configure will be set as the default LLM provider."
    echo ""
    
    if [ ! -f "$SECRETS_FILE" ]; then
        print_error "$SECRETS_FILE not found!"
        return 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "$CONFIG_FILE not found!"
        return 1
    fi
    
    local first_provider=""
    
    while true; do
        echo ""
        echo "  Select LLM provider to configure API key:"
        echo "    [1] google"
        echo "    [2] anthropic"
        echo "    [3] openai"
        echo ""
        
        read -p "Select provider (1-3, required): " SELECTION
        
        if [[ ! "$SELECTION" =~ ^[0-9]+$ ]] || [ "$SELECTION" -lt 1 ] || [ "$SELECTION" -gt 3 ]; then
            print_error "Invalid selection. Please enter 1, 2, or 3."
            continue
        fi
        
        local provider=""
        case $SELECTION in
            1) provider="google" ;;
            2) provider="anthropic" ;;
            3) provider="openai" ;;
        esac
        
        echo ""
        # Capitalize first letter for display (portable method)
        local provider_display=""
        case $provider in
            google) provider_display="Google" ;;
            anthropic) provider_display="Anthropic" ;;
            openai) provider_display="OpenAI" ;;
        esac
        read -p "Enter $provider_display API key: " api_key
        
        if [ -z "$api_key" ]; then
            print_error "API key cannot be empty. Please try again."
            continue
        fi
        
        # Handle OpenAI special case (base_url)
        if [ "$provider" = "openai" ]; then
            read -p "Enter OpenAI base_url (optional, for custom endpoints, press Enter to skip): " base_url
            
            python3 << EOF
import yaml
with open("$SECRETS_FILE", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
if 'openai' not in config:
    config['openai'] = {}
config['openai']['api_key'] = "$api_key"
if "$base_url":
    config['openai']['base_url'] = "$base_url"
with open("$SECRETS_FILE", 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
EOF
            print_success "Set OpenAI API key"
        else
            python3 << EOF
import yaml
with open("$SECRETS_FILE", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
if '$provider' not in config:
    config['$provider'] = {}
config['$provider']['api_key'] = "$api_key"
with open("$SECRETS_FILE", 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
EOF
            print_success "Set $provider_display API key"
        fi
        
        # Set the first provider as the default LLM provider
        if [ -z "$first_provider" ]; then
            first_provider="$provider"
            set_yaml_value "$CONFIG_FILE" "llm_provider" "$provider"
            print_success "Set default LLM provider to: $provider"
        fi
        
        echo ""
        read -p "Configure another API key? (y/N): " configure_another
        if [[ ! "$configure_another" =~ ^[Yy]$ ]]; then
            break
        fi
    done
}


configure_document_segmentation() {
    local step_num=$1
    print_step "$step_num" "Configuring Document Segmentation (Optional)"
    
    echo ""
    read -p "Configure document segmentation? (y/N): " configure_seg
    if [[ ! "$configure_seg" =~ ^[Yy]$ ]]; then
        print_info "Skipping document segmentation configuration"
        return 0
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "$CONFIG_FILE not found!"
        return 1
    fi
    
    echo ""
    echo "  Document segmentation options:"
    echo "    [1] enabled"
    echo "    [2] disabled"
    echo ""
    
    read -p "Select option (1-2): " SELECTION
    
    local enabled="false"
    if [[ "$SELECTION" =~ ^[0-9]+$ ]] && [ "$SELECTION" -eq 1 ]; then
        enabled="true"
    fi
    
    set_yaml_value "$CONFIG_FILE" "document_segmentation.enabled" "$enabled"
    print_success "Set document_segmentation.enabled to: $enabled"
    
    if [ "$enabled" = "true" ]; then
        read -p "Enter size threshold in characters (default: 50000): " threshold
        threshold=${threshold:-50000}
        set_yaml_value "$CONFIG_FILE" "document_segmentation.size_threshold_chars" "$threshold"
        print_success "Set size_threshold_chars to: $threshold"
    fi
    
    return 0
}

show_summary() {
    local step_num=$1
    print_step "$step_num" "Setup Summary"
    
    echo -e "\n  ${BOLD}Configuration Summary:${ENDC}\n"
    
    if [ -f "$CONFIG_FILE" ]; then
        local search_server=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')) or {}; print(c.get('default_search_server', 'not set'))" 2>/dev/null || echo "not set")
        local llm_provider=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')) or {}; print(c.get('llm_provider', 'not set'))" 2>/dev/null || echo "not set")
        local seg_enabled=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')) or {}; print(c.get('document_segmentation', {}).get('enabled', 'not set'))" 2>/dev/null || echo "not set")
        
        echo -e "  ${CYAN}Search Server:${ENDC} $search_server"
        echo -e "  ${CYAN}LLM Provider:${ENDC} $llm_provider"
        echo -e "  ${CYAN}Document Segmentation:${ENDC} $seg_enabled"
    fi
    
    if [ -f "$SECRETS_FILE" ]; then
        local has_openai=$(python3 -c "import yaml; c=yaml.safe_load(open('$SECRETS_FILE')) or {}; print('yes' if c.get('openai', {}).get('api_key') else 'no')" 2>/dev/null || echo "no")
        local has_anthropic=$(python3 -c "import yaml; c=yaml.safe_load(open('$SECRETS_FILE')) or {}; print('yes' if c.get('anthropic', {}).get('api_key') else 'no')" 2>/dev/null || echo "no")
        local has_google=$(python3 -c "import yaml; c=yaml.safe_load(open('$SECRETS_FILE')) or {}; print('yes' if c.get('google', {}).get('api_key') else 'no')" 2>/dev/null || echo "no")
        
        echo -e "\n  ${CYAN}API Keys Configured:${ENDC}"
        echo -e "    OpenAI: $has_openai"
        echo -e "    Anthropic: $has_anthropic"
        echo -e "    Google: $has_google"
    fi
    
    echo ""
}

main() {
    print_header
    
    echo -e "  ${BOLD}Welcome to DeepCode Setup!${ENDC}\n"
    echo "  This wizard will help you configure DeepCode automatically."
    echo "  Following the official installation recipe using UV package manager."
    echo ""
    echo "  Assuming you're already in the DeepCode directory."
    echo ""
    
    read -p "  Ready to begin setup? (Y/n): " begin
    if [[ "$begin" =~ ^[Nn]$ ]]; then
        echo "  Setup cancelled."
        exit 0
    fi
    
    # Track step number dynamically
    local step_counter=1
    
    # Step 1: Install UV
    install_uv $step_counter
    ((step_counter++))
    
    # Step 2: Setup venv with UV
    setup_venv $step_counter
    ((step_counter++))
    
    # Step 3: Install dependencies with UV
    install_dependencies $step_counter
    ((step_counter++))
    
    # Install Windows MCP servers (Windows only, before search server config)
    # This doesn't show a step number, it's just a prerequisite
    install_windows_mcp_servers
    
    # Step 4: Configure search server
    configure_search_server $step_counter
    ((step_counter++))
    
    # Step 5: Configure API keys (also sets default LLM provider)
    configure_api_keys $step_counter
    ((step_counter++))
    
    # Step 6: Configure Windows MCP servers (Windows only - configures paths)
    # Only increments counter if Windows (function returns early if not Windows)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        configure_windows_mcp_servers $step_counter
        ((step_counter++))
    fi
    
    # Step 7: Configure document segmentation (optional)
    configure_document_segmentation $step_counter
    ((step_counter++))
    
    # Step 8: Show summary
    show_summary $step_counter
    
    # Final message
    echo -e "\n  ${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${ENDC}"
    echo -e "  ${GREEN}${BOLD}║           ✅ Setup Complete! ✅                            ║${ENDC}"
    echo -e "  ${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${ENDC}\n"
    
    echo -e "  ${BOLD}Next steps:${ENDC}\n"
    echo -e "    1. Review configuration files:"
    echo -e "       - $CONFIG_FILE"
    echo -e "       - $SECRETS_FILE"
    echo ""
    echo -e "    2. Make sure your virtual environment is activated:"
    echo -e "       ${CYAN}source .venv/bin/activate${ENDC}  (Linux/macOS)"
    echo -e "       ${CYAN}.venv\\\\Scripts\\\\activate${ENDC}     (Windows)"
    echo ""
    echo -e "    3. Launch DeepCode:"
    echo -e "       ${CYAN}python deepcode.py${ENDC}          (Web interface)"
    echo -e "       ${CYAN}python cli/main_cli.py${ENDC}      (CLI interface)"
    echo ""
}

# Run main function
main "$@"
