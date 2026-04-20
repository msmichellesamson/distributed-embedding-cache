variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for compute resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for compute resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "cluster_size" {
  description = "Number of embedding cache nodes"
  type        = number
  default     = 3
}

variable "predictor_nodes" {
  description = "Number of embedding predictor nodes"
  type        = number
  default     = 2
}

variable "router_nodes" {
  description = "Number of similarity router nodes"
  type        = number
  default     = 2
}

# Service account for compute instances
resource "google_service_account" "embedding_cache_sa" {
  account_id   = "embedding-cache-${var.environment}"
  display_name = "Embedding Cache Service Account"
  description  = "Service account for distributed embedding cache instances"
}

resource "google_service_account_key" "embedding_cache_key" {
  service_account_id = google_service_account.embedding_cache_sa.name
}

# IAM roles for the service account
resource "google_project_iam_member" "cache_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.embedding_cache_sa.email}"
}

resource "google_project_iam_member" "cache_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.embedding_cache_sa.email}"
}

resource "google_project_iam_member" "cache_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.embedding_cache_sa.email}"
}

# Network configuration
data "google_compute_network" "default" {
  name = "default"
}

# Firewall rules for internal communication
resource "google_compute_firewall" "embedding_cache_internal" {
  name    = "embedding-cache-internal-${var.environment}"
  network = data.google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "50051", "9090", "6379", "16379"]
  }

  source_tags = ["embedding-cache-${var.environment}"]
  target_tags = ["embedding-cache-${var.environment}"]

  description = "Allow internal communication between embedding cache nodes"
}

# Firewall rule for external API access
resource "google_compute_firewall" "embedding_cache_external" {
  name    = "embedding-cache-external-${var.environment}"
  network = data.google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "50051"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["embedding-cache-${var.environment}"]

  description = "Allow external access to embedding cache API endpoints"
}

# Instance template for cache nodes
resource "google_compute_instance_template" "cache_node_template" {
  name_prefix  = "embedding-cache-node-${var.environment}-"
  machine_type = "n2-standard-4"
  region       = var.region

  tags = ["embedding-cache-${var.environment}", "cache-node"]

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2204-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
    disk_type    = "pd-ssd"
  }

  network_interface {
    network = data.google_compute_network.default.name
    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  metadata_startup_script = templatefile("${path.module}/scripts/cache_node_startup.sh", {
    service_account_key = base64decode(google_service_account_key.embedding_cache_key.private_key)
    project_id          = var.project_id
    environment         = var.environment
    node_type          = "cache"
  })

  service_account {
    email  = google_service_account.embedding_cache_sa.email
    scopes = ["cloud-platform"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Instance template for predictor nodes
resource "google_compute_instance_template" "predictor_node_template" {
  name_prefix  = "embedding-predictor-node-${var.environment}-"
  machine_type = "n2-standard-8"
  region       = var.region

  tags = ["embedding-cache-${var.environment}", "predictor-node"]

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2204-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 200
    disk_type    = "pd-ssd"
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  network_interface {
    network = data.google_compute_network.default.name
    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  metadata_startup_script = templatefile("${path.module}/scripts/predictor_node_startup.sh", {
    service_account_key = base64decode(google_service_account_key.embedding_cache_key.private_key)
    project_id          = var.project_id
    environment         = var.environment
    node_type          = "predictor"
  })

  service_account {
    email  = google_service_account.embedding_cache_sa.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    preemptible         = var.environment != "prod"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Instance template for router nodes
resource "google_compute_instance_template" "router_node_template" {
  name_prefix  = "embedding-router-node-${var.environment}-"
  machine_type = "n2-standard-2"
  region       = var.region

  tags = ["embedding-cache-${var.environment}", "router-node"]

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2204-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 50
    disk_type    = "pd-ssd"
  }

  network_interface {
    network = data.google_compute_network.default.name
    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  metadata_startup_script = templatefile("${path.module}/scripts/router_node_startup.sh", {
    service_account_key = base64decode(google_service_account_key.embedding_cache_key.private_key)
    project_id          = var.project_id
    environment         = var.environment
    node_type          = "router"
  })

  service_account {
    email  = google_service_account.embedding_cache_sa.email
    scopes = ["cloud-platform"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Managed instance groups
resource "google_compute_region_instance_group_manager" "cache_nodes" {
  name   = "embedding-cache-nodes-${var.environment}"
  region = var.region

  base_instance_name = "cache-node-${var.environment}"
  target_size        = var.cluster_size

  version {
    instance_template = google_compute_instance_template.cache_node_template.id
  }

  named_port {
    name = "http-api"
    port = 8000
  }

  named_port {
    name = "grpc-api"
    port = 50051
  }

  named_port {
    name = "metrics"
    port = 9090
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.cache_node_health.id
    initial_delay_sec = 300
  }
}

resource "google_compute_region_instance_group_manager" "predictor_nodes" {
  name   = "embedding-predictor-nodes-${var.environment}"
  region = var.region

  base_instance_name = "predictor-node-${var.environment}"
  target_size        = var.predictor_nodes

  version {
    instance_template = google_compute_instance_template.predictor_node_template.id
  }

  named_port {
    name = "grpc-api"
    port = 50052
  }

  named_port {
    name = "metrics"
    port = 9091
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.predictor_node_health.id
    initial_delay_sec = 600
  }
}

resource "google_compute_region_instance_group_manager" "router_nodes" {
  name   = "embedding-router-nodes-${var.environment}"
  region = var.region

  base_instance_name = "router-node-${var.environment}"
  target_size        = var.router_nodes

  version {
    instance_template = google_compute_instance_template.router_node_template.id
  }

  named_port {
    name = "grpc-api"
    port = 50053
  }

  named_port {
    name = "metrics"
    port = 9092
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.router_node_health.id
    initial_delay_sec = 300
  }
}

# Health checks
resource "google_compute_health_check" "cache_node_health" {
  name                = "embedding-cache-node-health-${var.environment}"
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = "8000"
    request_path = "/health"
  }
}

resource "google_compute_health_check" "predictor_node_health" {
  name                = "embedding-predictor-node-health-${var.environment}"
  check_interval_sec  = 60
  timeout_sec         = 15
  healthy_threshold   = 2
  unhealthy_threshold = 3

  tcp_health_check {
    port = "50052"
  }
}

resource "google_compute_health_check" "router_node_health" {
  name                = "embedding-router-node-health-${var.environment}"
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3

  tcp_health_check {
    port = "50053"
  }
}

# Load balancer for cache nodes
resource "google_compute_region_backend_service" "cache_backend" {
  name                  = "embedding-cache-backend-${var.environment}"
  region                = var.region
  load_balancing_scheme = "EXTERNAL"
  protocol              = "HTTP"
  timeout_sec           = 30

  backend {
    group = google_compute_region_instance_group_manager.cache_nodes.instance_group
  }

  health_checks = [google_compute_health_check.cache_node_health.id]

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

resource "google_compute_forwarding_rule" "cache_frontend" {
  name                  = "embedding-cache-frontend-${var.environment}"
  region                = var.region
  load_balancing_scheme = "EXTERNAL"
  port_range            = "8000"
  target                = google_compute_region_backend_service.cache_backend.id
}

# Autoscaling policies
resource "google_compute_region_autoscaler" "cache_autoscaler" {
  name   = "embedding-cache-autoscaler-${var.environment}"
  region = var.region
  target = google_compute_region_instance_group_manager.cache_nodes.id

  autoscaling_policy {
    max_replicas    = var.cluster_size * 2
    min_replicas    = var.cluster_size
    cooldown_period = 300

    cpu_utilization {
      target = 0.7
    }

    metric {
      name   = "custom.googleapis.com/embedding_cache/cache_hit_rate"
      target = 0.85
      type   = "GAUGE"
    }

    scale_in_control {
      max_scaled_in_replicas {
        fixed = 1
      }
      time_window_sec = 600
    }
  }
}

resource "google_compute_region_autoscaler" "predictor_autoscaler" {
  name   = "embedding-predictor-autoscaler-${var.environment}"
  region = var.region
  target = google_compute_region_instance_group_manager.predictor_nodes.id

  autoscaling_policy {
    max_replicas    = var.predictor_nodes * 3
    min_replicas    = var.predictor_nodes
    cooldown_period = 600

    cpu_utilization {
      target = 0.8
    }

    metric {
      name   = "custom.googleapis.com/embedding_cache/prediction_queue_length"
      target = 100
      type   = "GAUGE"
    }

    scale_in_control {
      max_scaled_in_replicas {
        fixed = 1
      }
      time_window_sec = 1200
    }
  }
}

# Outputs
output "cache_load_balancer_ip" {
  description = "External IP of the cache load balancer"
  value       = google_compute_forwarding_rule.cache_frontend.ip_address
}

output "cache_instance_group" {
  description = "Cache nodes instance group"
  value       = google_compute_region_instance_group_manager.cache_nodes.instance_group
}

output "predictor_instance_group" {
  description = "Predictor nodes instance group"
  value       = google_compute_region_instance_group_manager.predictor_nodes.instance_group
}

output "router_instance_group" {
  description = "Router nodes instance group"
  value       = google_compute_region_instance_group_manager.router_nodes.instance_group
}

output "service_account_email" {
  description = "Service account email for embedding cache instances"
  value       = google_service_account.embedding_cache_sa.email
}