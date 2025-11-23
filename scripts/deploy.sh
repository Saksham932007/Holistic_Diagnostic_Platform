#!/bin/bash

# Medical Platform Deployment Automation Script
# Comprehensive deployment with rollback capabilities and health monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_CONFIG="${PROJECT_ROOT}/config/deployment.conf"

# Load configuration if exists
if [[ -f "$DEPLOYMENT_CONFIG" ]]; then
    source "$DEPLOYMENT_CONFIG"
fi

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="${NAMESPACE:-medical-platform-${ENVIRONMENT}}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-medical-diagnostic-platform}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-600}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-30}"
ROLLBACK_ENABLED="${ROLLBACK_ENABLED:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Function to display usage
usage() {
    cat << EOF
Medical Platform Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy          Deploy application to specified environment
    rollback        Rollback to previous deployment
    status          Check deployment status
    logs            View application logs
    scale           Scale application replicas
    health          Check application health
    cleanup         Clean up old deployments

Options:
    -e, --environment ENV    Target environment (staging|production)
    -t, --tag TAG           Docker image tag to deploy
    -n, --namespace NS      Kubernetes namespace
    -r, --replicas NUM      Number of replicas to deploy
    -f, --force             Force deployment without confirmation
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0 -e staging deploy
    $0 -e production -t v1.2.0 deploy
    $0 -e staging rollback
    $0 -e production scale -r 5

EOF
}

# Parse command line arguments
FORCE=false
VERBOSE=false
REPLICAS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            NAMESPACE="medical-platform-${ENVIRONMENT}"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--replicas)
            REPLICAS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy|rollback|status|logs|scale|health|cleanup)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command specified"
    usage
    exit 1
fi

# Verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        staging|production)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Use 'staging' or 'production'"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm (if using Helm)
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed (optional)"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Verify kubectl context
    CURRENT_CONTEXT=$(kubectl config current-context)
    log_info "Current kubectl context: $CURRENT_CONTEXT"
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Prerequisites check completed"
}

# Build and push Docker image
build_and_push_image() {
    local full_image_name="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Building Docker image: $full_image_name"
    
    cd "$PROJECT_ROOT"
    
    # Build image
    docker build \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from "${IMAGE_REGISTRY}/${IMAGE_NAME}:latest" \
        -t "$full_image_name" \
        .
    
    # Tag as latest if not already
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        docker tag "$full_image_name" "${IMAGE_REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    # Push image
    log_info "Pushing Docker image: $full_image_name"
    docker push "$full_image_name"
    
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        docker push "${IMAGE_REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    log_success "Docker image built and pushed successfully"
}

# Backup current deployment
backup_deployment() {
    local backup_dir="${PROJECT_ROOT}/backups/${ENVIRONMENT}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${backup_dir}/deployment_${timestamp}.yaml"
    
    mkdir -p "$backup_dir"
    
    log_info "Backing up current deployment..."
    
    # Backup deployments
    kubectl get deployments -n "$NAMESPACE" -o yaml > "$backup_file"
    
    # Store backup info
    echo "${timestamp}" > "${backup_dir}/latest_backup"
    
    log_success "Deployment backed up to: $backup_file"
}

# Deploy application
deploy_application() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    
    # Confirm deployment in production
    if [[ "$ENVIRONMENT" == "production" && "$FORCE" != "true" ]]; then
        echo -n "Are you sure you want to deploy to production? (y/N): "
        read -r confirmation
        if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Build and push image if needed
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        build_and_push_image
    fi
    
    # Backup current deployment
    backup_deployment
    
    # Update image in Kubernetes manifests
    local full_image_name="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f "${PROJECT_ROOT}/k8s/" -n "$NAMESPACE"
    
    # Update image in deployment
    kubectl set image deployment/medical-api \
        medical-api="$full_image_name" \
        -n "$NAMESPACE"
    
    # Scale replicas if specified
    if [[ -n "$REPLICAS" ]]; then
        log_info "Scaling to $REPLICAS replicas..."
        kubectl scale deployment/medical-api \
            --replicas="$REPLICAS" \
            -n "$NAMESPACE"
    fi
    
    # Wait for rollout to complete
    log_info "Waiting for deployment to complete..."
    kubectl rollout status deployment/medical-api \
        -n "$NAMESPACE" \
        --timeout="${KUBECTL_TIMEOUT}s"
    
    log_success "Deployment completed successfully"
    
    # Verify deployment
    verify_deployment
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment health..."
    
    # Wait for pods to be ready
    local retries=0
    while [[ $retries -lt $HEALTH_CHECK_RETRIES ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" \
            -l app=medical-api \
            -o jsonpath='{.items[*].status.containerStatuses[0].ready}' \
            | tr ' ' '\n' | grep -c true || true)
        
        local total_pods=$(kubectl get pods -n "$NAMESPACE" \
            -l app=medical-api \
            -o jsonpath='{.items[*].metadata.name}' \
            | wc -w)
        
        if [[ "$ready_pods" -eq "$total_pods" && "$total_pods" -gt 0 ]]; then
            log_success "All $total_pods pods are ready"
            break
        fi
        
        log_info "Waiting for pods to be ready ($ready_pods/$total_pods)..."
        sleep 10
        ((retries++))
    done
    
    if [[ $retries -eq $HEALTH_CHECK_RETRIES ]]; then
        log_error "Health check timeout. Some pods are not ready"
        
        # Show pod status for debugging
        kubectl get pods -n "$NAMESPACE" -l app=medical-api
        
        if [[ "$ROLLBACK_ENABLED" == "true" ]]; then
            log_warning "Initiating automatic rollback..."
            rollback_deployment
        fi
        exit 1
    fi
    
    # Test application health endpoint
    test_application_health
}

# Test application health
test_application_health() {
    log_info "Testing application health endpoint..."
    
    # Get service endpoint
    local service_url
    if [[ "$ENVIRONMENT" == "production" ]]; then
        service_url="https://api.medical-platform.com"
    else
        service_url="https://staging.api.medical-platform.com"
    fi
    
    # Port forward for local testing if needed
    if [[ "$service_url" == "local" ]]; then
        kubectl port-forward service/medical-api-service 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        service_url="http://localhost:8080"
        sleep 5
    fi
    
    # Test health endpoint
    local retries=0
    while [[ $retries -lt 10 ]]; do
        if curl -f -s "${service_url}/health" > /dev/null; then
            log_success "Application health check passed"
            
            # Clean up port forward
            if [[ -n "${port_forward_pid:-}" ]]; then
                kill "$port_forward_pid" 2>/dev/null || true
            fi
            
            return 0
        fi
        
        log_info "Health endpoint not responding, retrying..."
        sleep 10
        ((retries++))
    done
    
    # Clean up port forward
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "$port_forward_pid" 2>/dev/null || true
    fi
    
    log_error "Application health check failed"
    return 1
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    # Get previous revision
    local previous_revision=$(kubectl rollout history deployment/medical-api -n "$NAMESPACE" \
        | tail -2 | head -1 | awk '{print $1}')
    
    if [[ -z "$previous_revision" ]]; then
        log_error "No previous revision found for rollback"
        exit 1
    fi
    
    # Perform rollback
    kubectl rollout undo deployment/medical-api \
        --to-revision="$previous_revision" \
        -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/medical-api \
        -n "$NAMESPACE" \
        --timeout="${KUBECTL_TIMEOUT}s"
    
    # Verify rollback
    verify_deployment
    
    log_success "Rollback completed successfully"
}

# Get deployment status
get_deployment_status() {
    log_info "Getting deployment status for $ENVIRONMENT environment..."
    
    echo "=== Deployment Status ==="
    kubectl get deployments -n "$NAMESPACE"
    
    echo -e "\n=== Pod Status ==="
    kubectl get pods -n "$NAMESPACE" -l app=medical-api
    
    echo -e "\n=== Service Status ==="
    kubectl get services -n "$NAMESPACE"
    
    echo -e "\n=== Ingress Status ==="
    kubectl get ingress -n "$NAMESPACE"
    
    echo -e "\n=== Recent Events ==="
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# View application logs
view_logs() {
    log_info "Viewing logs for $ENVIRONMENT environment..."
    
    # Get pod names
    local pods=$(kubectl get pods -n "$NAMESPACE" \
        -l app=medical-api \
        -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$pods" ]]; then
        log_error "No pods found"
        exit 1
    fi
    
    # Follow logs from all pods
    kubectl logs -f deployment/medical-api -n "$NAMESPACE" --all-containers=true
}

# Scale deployment
scale_deployment() {
    if [[ -z "$REPLICAS" ]]; then
        log_error "Number of replicas not specified. Use -r option"
        exit 1
    fi
    
    log_info "Scaling deployment to $REPLICAS replicas..."
    
    kubectl scale deployment/medical-api \
        --replicas="$REPLICAS" \
        -n "$NAMESPACE"
    
    # Wait for scaling to complete
    kubectl rollout status deployment/medical-api \
        -n "$NAMESPACE" \
        --timeout="${KUBECTL_TIMEOUT}s"
    
    log_success "Scaling completed successfully"
}

# Cleanup old deployments
cleanup_deployments() {
    log_info "Cleaning up old deployments and resources..."
    
    # Keep only last 5 replica sets
    local replica_sets=$(kubectl get rs -n "$NAMESPACE" \
        -l app=medical-api \
        --sort-by='.metadata.creationTimestamp' \
        -o jsonpath='{.items[*].metadata.name}')
    
    local rs_array=($replica_sets)
    local rs_count=${#rs_array[@]}
    
    if [[ $rs_count -gt 5 ]]; then
        local rs_to_delete=${rs_array[@]:0:$((rs_count-5))}
        for rs in $rs_to_delete; do
            log_info "Deleting old replica set: $rs"
            kubectl delete rs "$rs" -n "$NAMESPACE"
        done
    fi
    
    # Clean up old backup files (keep last 10)
    local backup_dir="${PROJECT_ROOT}/backups/${ENVIRONMENT}"
    if [[ -d "$backup_dir" ]]; then
        find "$backup_dir" -name "deployment_*.yaml" -type f \
            | sort | head -n -10 | xargs -r rm
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    validate_environment
    check_prerequisites
    
    case "$COMMAND" in
        deploy)
            deploy_application
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            get_deployment_status
            ;;
        logs)
            view_logs
            ;;
        scale)
            scale_deployment
            ;;
        health)
            test_application_health
            ;;
        cleanup)
            cleanup_deployments
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Run main function
main