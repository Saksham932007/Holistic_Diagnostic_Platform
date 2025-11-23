#!/bin/bash

"""
Comprehensive System Health Check Script

Advanced health monitoring for all platform components
including AI models, database, storage, and services.
"""

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/tmp/health_check_${TIMESTAMP}.log"
HEALTH_STATUS_FILE="/tmp/platform_health_status.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
HEALTH_RESULTS=()
OVERALL_STATUS="HEALTHY"
CRITICAL_ISSUES=()
WARNING_ISSUES=()

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored status
print_status() {
    local status=$1
    local component=$2
    local message=$3
    
    case $status in
        "HEALTHY")
            echo -e "${GREEN}[✓]${NC} $component: $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[⚠]${NC} $component: $message"
            WARNING_ISSUES+=("$component: $message")
            ;;
        "CRITICAL")
            echo -e "${RED}[✗]${NC} $component: $message"
            CRITICAL_ISSUES+=("$component: $message")
            OVERALL_STATUS="CRITICAL"
            ;;
        "DEGRADED")
            echo -e "${YELLOW}[△]${NC} $component: $message"
            WARNING_ISSUES+=("$component: $message")
            if [[ "$OVERALL_STATUS" == "HEALTHY" ]]; then
                OVERALL_STATUS="DEGRADED"
            fi
            ;;
    esac
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Memory check
    local memory_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
    local memory_usage_int=${memory_usage%.*}
    
    if [ "$memory_usage_int" -gt 90 ]; then
        print_status "CRITICAL" "Memory" "Usage at ${memory_usage}%"
    elif [ "$memory_usage_int" -gt 75 ]; then
        print_status "WARNING" "Memory" "Usage at ${memory_usage}%"
    else
        print_status "HEALTHY" "Memory" "Usage at ${memory_usage}%"
    fi
    
    # CPU check
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    local cpu_usage_int=${cpu_usage%.*}
    
    if [ "$cpu_usage_int" -gt 90 ]; then
        print_status "CRITICAL" "CPU" "Usage at ${cpu_usage}%"
    elif [ "$cpu_usage_int" -gt 75 ]; then
        print_status "WARNING" "CPU" "Usage at ${cpu_usage}%"
    else
        print_status "HEALTHY" "CPU" "Usage at ${cpu_usage}%"
    fi
    
    # Disk space check
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        print_status "CRITICAL" "Disk" "Usage at ${disk_usage}%"
    elif [ "$disk_usage" -gt 80 ]; then
        print_status "WARNING" "Disk" "Usage at ${disk_usage}%"
    else
        print_status "HEALTHY" "Disk" "Usage at ${disk_usage}%"
    fi
    
    # Load average check
    local load_avg=$(uptime | awk '{print $10}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local load_ratio=$(echo "scale=2; $load_avg / $cpu_cores" | bc)
    
    if (( $(echo "$load_ratio > 2.0" | bc -l) )); then
        print_status "CRITICAL" "Load" "Average ${load_avg} on ${cpu_cores} cores"
    elif (( $(echo "$load_ratio > 1.5" | bc -l) )); then
        print_status "WARNING" "Load" "Average ${load_avg} on ${cpu_cores} cores"
    else
        print_status "HEALTHY" "Load" "Average ${load_avg} on ${cpu_cores} cores"
    fi
}

# Check Python environment
check_python_environment() {
    log "Checking Python environment..."
    
    # Python version check
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1)
        print_status "HEALTHY" "Python" "$python_version available"
    else
        print_status "CRITICAL" "Python" "Python3 not found"
        return
    fi
    
    # Virtual environment check
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_status "HEALTHY" "Virtual Env" "Active: $VIRTUAL_ENV"
    else
        print_status "WARNING" "Virtual Env" "No virtual environment active"
    fi
    
    # Required packages check
    local required_packages=("torch" "torchvision" "monai" "fastapi" "sqlalchemy" "numpy" "pandas")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_status "HEALTHY" "Python Packages" "All required packages available"
    else
        print_status "CRITICAL" "Python Packages" "Missing: ${missing_packages[*]}"
    fi
}

# Check GPU availability
check_gpu() {
    log "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
        local gpu_count=$(echo "$gpu_info" | wc -l)
        
        print_status "HEALTHY" "GPU Hardware" "$gpu_count GPU(s) detected"
        
        # Check GPU memory usage
        while IFS=',' read -r name mem_used mem_total gpu_util; do
            local mem_percent=$((mem_used * 100 / mem_total))
            
            if [ "$mem_percent" -gt 90 ]; then
                print_status "WARNING" "GPU Memory" "$name: ${mem_percent}% used"
            else
                print_status "HEALTHY" "GPU Memory" "$name: ${mem_percent}% used"
            fi
            
            # Check GPU utilization
            if [ "$gpu_util" -gt 95 ]; then
                print_status "WARNING" "GPU Utilization" "$name: ${gpu_util}%"
            else
                print_status "HEALTHY" "GPU Utilization" "$name: ${gpu_util}%"
            fi
        done <<< "$gpu_info"
        
        # Check CUDA availability
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_status "HEALTHY" "CUDA" "Available and functional"
        else
            print_status "CRITICAL" "CUDA" "Not available or not functional"
        fi
    else
        print_status "WARNING" "GPU" "No NVIDIA GPU detected or nvidia-smi not available"
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    # Check if we can connect to database
    local db_check_script="
import sys
import os
sys.path.append('$PROJECT_ROOT')
try:
    from src.core.config import settings
    from sqlalchemy import create_engine, text
    
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"
    
    if python3 -c "$db_check_script" 2>/dev/null; then
        print_status "HEALTHY" "Database" "Connection successful"
    else
        print_status "CRITICAL" "Database" "Connection failed"
    fi
    
    # Check database size if PostgreSQL
    local db_size_script="
import sys
sys.path.append('$PROJECT_ROOT')
try:
    from src.core.config import settings
    from sqlalchemy import create_engine, text
    
    if 'postgresql' in settings.database_url:
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            result = conn.execute(text(\"SELECT pg_size_pretty(pg_database_size(current_database()))\"))
            size = result.fetchone()[0]
            print(f'Database size: {size}')
    else:
        print('Database size check not applicable for this database type')
except Exception as e:
    print(f'Database size check failed: {e}')
"
    
    python3 -c "$db_size_script" 2>/dev/null || true
}

# Check model files
check_model_files() {
    log "Checking model files..."
    
    local model_dirs=("$PROJECT_ROOT/models" "$PROJECT_ROOT/checkpoints")
    local model_files_found=0
    local total_model_size=0
    
    for model_dir in "${model_dirs[@]}"; do
        if [ -d "$model_dir" ]; then
            local file_count=$(find "$model_dir" -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" | wc -l)
            model_files_found=$((model_files_found + file_count))
            
            # Calculate total size
            if [ "$file_count" -gt 0 ]; then
                local dir_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
                print_status "HEALTHY" "Model Files" "$file_count files in $model_dir ($dir_size)"
            fi
        fi
    done
    
    if [ "$model_files_found" -eq 0 ]; then
        print_status "WARNING" "Model Files" "No model files found"
    fi
}

# Check API service
check_api_service() {
    log "Checking API service..."
    
    # Check if FastAPI app can be imported
    local api_import_script="
import sys
sys.path.append('$PROJECT_ROOT')
try:
    from src.api.main import app
    print('FastAPI app import successful')
except Exception as e:
    print(f'FastAPI app import failed: {e}')
    sys.exit(1)
"
    
    if python3 -c "$api_import_script" 2>/dev/null; then
        print_status "HEALTHY" "API Service" "FastAPI app can be imported"
    else
        print_status "CRITICAL" "API Service" "FastAPI app import failed"
    fi
    
    # Check if service is running (try common ports)
    local ports=(8000 8080 5000)
    local service_running=false
    
    for port in "${ports[@]}"; do
        if netstat -tulpn 2>/dev/null | grep -q ":$port "; then
            print_status "HEALTHY" "API Service" "Service running on port $port"
            service_running=true
            break
        fi
    done
    
    if [ "$service_running" = false ]; then
        print_status "WARNING" "API Service" "No service detected on common ports"
    fi
}

# Check storage directories
check_storage() {
    log "Checking storage directories..."
    
    local storage_dirs=("data" "logs" "uploads" "outputs" "backups")
    
    for dir_name in "${storage_dirs[@]}"; do
        local dir_path="$PROJECT_ROOT/$dir_name"
        
        if [ -d "$dir_path" ]; then
            local permissions=$(stat -c "%a" "$dir_path")
            local size=$(du -sh "$dir_path" 2>/dev/null | cut -f1)
            local file_count=$(find "$dir_path" -type f | wc -l)
            
            print_status "HEALTHY" "Storage ($dir_name)" "$file_count files, $size total ($permissions permissions)"
        else
            print_status "WARNING" "Storage ($dir_name)" "Directory does not exist: $dir_path"
        fi
    done
    
    # Check available space in common mount points
    local mount_points=("/" "/tmp" "/var")
    
    for mount_point in "${mount_points[@]}"; do
        if [ -d "$mount_point" ]; then
            local available=$(df -h "$mount_point" | tail -1 | awk '{print $4}')
            print_status "HEALTHY" "Storage Space ($mount_point)" "$available available"
        fi
    done
}

# Check network connectivity
check_network() {
    log "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        print_status "HEALTHY" "Internet" "Connectivity available"
    else
        print_status "WARNING" "Internet" "No internet connectivity"
    fi
    
    # Check DNS resolution
    if nslookup google.com &> /dev/null; then
        print_status "HEALTHY" "DNS" "Resolution working"
    else
        print_status "WARNING" "DNS" "Resolution issues"
    fi
    
    # Check if firewall is active
    if command -v ufw &> /dev/null; then
        local ufw_status=$(ufw status | head -1)
        print_status "HEALTHY" "Firewall" "$ufw_status"
    elif command -v iptables &> /dev/null; then
        local iptables_rules=$(iptables -L | wc -l)
        print_status "HEALTHY" "Firewall" "$iptables_rules iptables rules active"
    else
        print_status "WARNING" "Firewall" "No firewall management tools found"
    fi
}

# Check security configurations
check_security() {
    log "Checking security configurations..."
    
    # Check file permissions on sensitive files
    local sensitive_files=(
        "$PROJECT_ROOT/src/core/config.py"
        "$PROJECT_ROOT/.env"
        "$PROJECT_ROOT/keys"
    )
    
    for file_path in "${sensitive_files[@]}"; do
        if [ -f "$file_path" ] || [ -d "$file_path" ]; then
            local permissions=$(stat -c "%a" "$file_path" 2>/dev/null || echo "unknown")
            
            case "$permissions" in
                "600"|"700")
                    print_status "HEALTHY" "Security ($(basename "$file_path"))" "Secure permissions ($permissions)"
                    ;;
                "644"|"755")
                    print_status "WARNING" "Security ($(basename "$file_path"))" "Potentially insecure permissions ($permissions)"
                    ;;
                *)
                    print_status "WARNING" "Security ($(basename "$file_path"))" "Check permissions ($permissions)"
                    ;;
            esac
        fi
    done
    
    # Check for running processes
    local sensitive_processes=("sshd" "nginx" "apache2")
    
    for process in "${sensitive_processes[@]}"; do
        if pgrep -x "$process" > /dev/null; then
            print_status "HEALTHY" "Security ($process)" "Service running"
        fi
    done
}

# Check log files for errors
check_logs() {
    log "Checking recent log entries..."
    
    local log_dirs=("$PROJECT_ROOT/logs" "/var/log" "/tmp")
    local error_count=0
    local warning_count=0
    
    for log_dir in "${log_dirs[@]}"; do
        if [ -d "$log_dir" ]; then
            # Check for recent errors (last 24 hours)
            local recent_errors=$(find "$log_dir" -name "*.log" -mtime -1 -exec grep -i "error\|critical\|fatal" {} \; 2>/dev/null | wc -l)
            local recent_warnings=$(find "$log_dir" -name "*.log" -mtime -1 -exec grep -i "warning\|warn" {} \; 2>/dev/null | wc -l)
            
            error_count=$((error_count + recent_errors))
            warning_count=$((warning_count + recent_warnings))
        fi
    done
    
    if [ "$error_count" -gt 10 ]; then
        print_status "WARNING" "Log Analysis" "$error_count errors in last 24 hours"
    elif [ "$error_count" -gt 0 ]; then
        print_status "HEALTHY" "Log Analysis" "$error_count errors, $warning_count warnings in last 24 hours"
    else
        print_status "HEALTHY" "Log Analysis" "No significant errors in recent logs"
    fi
}

# Generate health report
generate_health_report() {
    local json_report=$(cat << EOF
{
    "timestamp": "$(date -Iseconds)",
    "overall_status": "$OVERALL_STATUS",
    "critical_issues": [
        $(printf '"%s",' "${CRITICAL_ISSUES[@]}" | sed 's/,$//')
    ],
    "warning_issues": [
        $(printf '"%s",' "${WARNING_ISSUES[@]}" | sed 's/,$//')
    ],
    "health_check_log": "$LOG_FILE",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "uptime": "$(uptime -p)",
        "users": $(who | wc -l)
    }
}
EOF
)
    
    echo "$json_report" > "$HEALTH_STATUS_FILE"
    log "Health report saved to: $HEALTH_STATUS_FILE"
}

# Print summary
print_summary() {
    echo -e "\n${BLUE}=== HEALTH CHECK SUMMARY ===${NC}"
    echo -e "Overall Status: ${GREEN}$OVERALL_STATUS${NC}"
    echo -e "Timestamp: $(date)"
    
    if [ ${#CRITICAL_ISSUES[@]} -gt 0 ]; then
        echo -e "\n${RED}Critical Issues:${NC}"
        printf '%s\n' "${CRITICAL_ISSUES[@]}"
    fi
    
    if [ ${#WARNING_ISSUES[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}Warnings:${NC}"
        printf '%s\n' "${WARNING_ISSUES[@]}"
    fi
    
    echo -e "\nDetailed log: $LOG_FILE"
    echo -e "Health status: $HEALTH_STATUS_FILE"
}

# Main execution
main() {
    echo -e "${BLUE}=== HOLISTIC DIAGNOSTIC PLATFORM HEALTH CHECK ===${NC}"
    echo -e "Starting comprehensive health check at $(date)\n"
    
    # Run all health checks
    check_system_resources
    echo
    check_python_environment
    echo
    check_gpu
    echo
    check_database
    echo
    check_model_files
    echo
    check_api_service
    echo
    check_storage
    echo
    check_network
    echo
    check_security
    echo
    check_logs
    
    # Generate report and summary
    generate_health_report
    print_summary
}

# Run health check if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi