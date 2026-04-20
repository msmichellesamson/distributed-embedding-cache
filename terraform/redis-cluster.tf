variable "project_id" {
  description = "GCP project ID for Redis cluster deployment"
  type        = string
}

variable "region" {
  description = "GCP region for Redis cluster"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "redis_cluster_size" {
  description = "Number of Redis nodes in cluster"
  type        = number
  default     = 6
  validation {
    condition     = var.redis_cluster_size >= 3 && var.redis_cluster_size % 2 == 0
    error_message = "Redis cluster size must be at least 3 and an even number for proper master-slave distribution."
  }
}

variable "redis_memory_size_gb" {
  description = "Memory size per Redis node in GB"
  type        = number
  default     = 8
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

variable "authorized_networks" {
  description = "List of authorized networks for Redis access"
  type = list(object({
    value         = string
    display_name  = string
  }))
  default = [
    {
      value        = "10.0.0.0/8"
      display_name = "private-networks"
    }
  ]
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}

locals {
  cluster_name = "embedding-cache-${var.environment}"
  labels = {
    environment = var.environment
    component   = "embedding-cache"
    managed_by  = "terraform"
  }
}

resource "google_redis_cluster" "embedding_cache" {
  name               = local.cluster_name
  project            = var.project_id
  region             = var.region
  shard_count        = var.redis_cluster_size / 2
  replica_count      = 1
  node_type         = "redis-standard-small"
  transit_encryption_mode = "TRANSIT_ENCRYPTION_MODE_SERVER_AUTHENTICATION"
  authorization_mode = "AUTH_MODE_REDIS_AUTH"
  
  psc_configs {
    network = google_compute_network.redis_network.id
  }

  depends_on = [
    google_compute_network.redis_network,
    google_compute_subnetwork.redis_subnet
  ]
}

resource "google_compute_network" "redis_network" {
  name                    = "${local.cluster_name}-network"
  project                 = var.project_id
  auto_create_subnetworks = false
  mtu                     = 1500
  
  description = "VPC network for Redis embedding cache cluster"
}

resource "google_compute_subnetwork" "redis_subnet" {
  name          = "${local.cluster_name}-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.redis_network.id
  ip_cidr_range = "10.0.0.0/24"
  
  description = "Subnet for Redis embedding cache cluster"
  
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "pod-ranges"
    ip_cidr_range = "10.2.0.0/16"
  }
}

resource "google_compute_firewall" "redis_internal" {
  name    = "${local.cluster_name}-internal"
  project = var.project_id
  network = google_compute_network.redis_network.name

  allow {
    protocol = "tcp"
    ports    = ["6379", "16379"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["redis-cluster"]
  
  description = "Allow internal Redis cluster communication"
}

resource "google_compute_firewall" "redis_health_checks" {
  name    = "${local.cluster_name}-health-checks"
  project = var.project_id
  network = google_compute_network.redis_network.name

  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]
  target_tags = ["redis-cluster"]
  
  description = "Allow Google Cloud health checks for Redis"
}

resource "google_redis_instance" "redis_metrics" {
  name               = "${local.cluster_name}-metrics"
  project            = var.project_id
  region             = var.region
  memory_size_gb     = 1
  redis_version      = var.redis_version
  display_name       = "Redis metrics store for embedding cache"
  
  authorized_network = google_compute_network.redis_network.id
  
  labels = merge(local.labels, {
    role = "metrics"
  })
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout          = "300"
    tcp-keepalive    = "60"
  }
}

resource "google_monitoring_alert_policy" "redis_memory_usage" {
  project      = var.project_id
  display_name = "Redis Cluster High Memory Usage"
  combiner     = "OR"
  
  conditions {
    display_name = "Memory usage > 80%"
    condition_threshold {
      filter          = "resource.type=\"redis_instance\" AND resource.label.instance_id=\"${google_redis_cluster.embedding_cache.name}\""
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8
      duration        = "300s"
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "redis_connection_count" {
  project      = var.project_id
  display_name = "Redis Cluster High Connection Count"
  combiner     = "OR"
  
  conditions {
    display_name = "Connection count > 1000"
    condition_threshold {
      filter          = "resource.type=\"redis_instance\" AND resource.label.instance_id=\"${google_redis_cluster.embedding_cache.name}\""
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 1000
      duration        = "180s"
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_compute_instance_group_manager" "redis_proxy" {
  name               = "${local.cluster_name}-proxy"
  project            = var.project_id
  zone               = "${var.region}-a"
  base_instance_name = "redis-proxy"
  
  version {
    instance_template = google_compute_instance_template.redis_proxy.id
  }
  
  target_size = 2
  
  named_port {
    name = "redis-proxy"
    port = 8000
  }
  
  auto_healing_policies {
    health_check      = google_compute_health_check.redis_proxy.id
    initial_delay_sec = 300
  }
}

resource "google_compute_instance_template" "redis_proxy" {
  name_prefix  = "${local.cluster_name}-proxy-"
  project      = var.project_id
  machine_type = "e2-standard-2"
  region       = var.region
  
  disk {
    source_image = "cos-cloud/cos-stable"
    auto_delete  = true
    boot         = true
    disk_size_gb = 20
    type         = "pd-standard"
  }
  
  network_interface {
    network    = google_compute_network.redis_network.id
    subnetwork = google_compute_subnetwork.redis_subnet.id
  }
  
  metadata = {
    startup-script = templatefile("${path.module}/startup-script.sh", {
      redis_cluster_endpoint = google_redis_cluster.embedding_cache.psc_connections[0].address
    })
  }
  
  service_account {
    email = google_service_account.redis_proxy.email
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/logging.write"
    ]
  }
  
  tags = ["redis-proxy", "redis-cluster"]
  
  labels = merge(local.labels, {
    component = "redis-proxy"
  })
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_health_check" "redis_proxy" {
  name    = "${local.cluster_name}-proxy-health"
  project = var.project_id
  
  timeout_sec         = 5
  check_interval_sec  = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
  
  tcp_health_check {
    port = "8000"
  }
}

resource "google_service_account" "redis_proxy" {
  account_id   = "${local.cluster_name}-proxy"
  project      = var.project_id
  display_name = "Service account for Redis proxy instances"
}

resource "google_project_iam_member" "redis_proxy_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.redis_proxy.email}"
}

resource "google_project_iam_member" "redis_proxy_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.redis_proxy.email}"
}

resource "google_compute_backend_service" "redis_proxy" {
  name                  = "${local.cluster_name}-proxy-backend"
  project               = var.project_id
  protocol              = "TCP"
  load_balancing_scheme = "INTERNAL"
  health_checks         = [google_compute_health_check.redis_proxy.id]
  
  backend {
    group                 = google_compute_instance_group_manager.redis_proxy.instance_group
    balancing_mode        = "CONNECTION"
    max_connections       = 1000
  }
  
  connection_draining_timeout_sec = 30
}

resource "google_compute_forwarding_rule" "redis_proxy" {
  name                  = "${local.cluster_name}-proxy-lb"
  project               = var.project_id
  region                = var.region
  load_balancing_scheme = "INTERNAL"
  backend_service       = google_compute_backend_service.redis_proxy.id
  ip_protocol           = "TCP"
  port_range            = "8000"
  network               = google_compute_network.redis_network.id
  subnetwork            = google_compute_subnetwork.redis_subnet.id
}

output "redis_cluster_endpoint" {
  description = "Redis cluster PSC endpoint"
  value       = google_redis_cluster.embedding_cache.psc_connections[0].address
  sensitive   = false
}

output "redis_cluster_port" {
  description = "Redis cluster port"
  value       = 6379
}

output "redis_metrics_host" {
  description = "Redis metrics instance host"
  value       = google_redis_instance.redis_metrics.host
}

output "redis_metrics_port" {
  description = "Redis metrics instance port"
  value       = google_redis_instance.redis_metrics.port
}

output "redis_proxy_ip" {
  description = "Internal load balancer IP for Redis proxy"
  value       = google_compute_forwarding_rule.redis_proxy.ip_address
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.redis_network.name
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.redis_subnet.name
}

output "cluster_auth_string" {
  description = "Redis cluster auth string"
  value       = google_redis_cluster.embedding_cache.auth_string
  sensitive   = true
}