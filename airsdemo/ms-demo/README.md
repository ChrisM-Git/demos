# Kubernetes App Template

A GitLab project template designed to simplify building and deploying Dockerized applications to Google Kubernetes Engine (GKE).

## Overview

This template provides a pre-configured CI/CD pipeline that automates the entire deployment lifecycle for containerized applications. It includes automated building, testing, and deployment to Kubernetes clusters with support for merge request preview environments and production deployments.

## Features

- **Automated Docker Build**: Automatically builds Docker images from your Dockerfile on every commit
- **Kubernetes Deployment**: Deploys containerized applications to GKE with minimal configuration
- **Preview Environments**: Creates isolated preview environments for merge requests
- **Production Deployment**: Automatic deployment to production on main branch merges
- **Environment Cleanup**: Manual cleanup jobs for merge request preview environments

## Getting Started

1. **Set up infrastructure**: Use the [airs/demo-env/infra](https://code.pan.run/airs/demo-env/infra) repository to create required Google Cloud resources, including service accounts for workload identity
2. **Add your Dockerfile**: Create a `Dockerfile` in the root of your project
3. **Configure variables**: Set required environment variables for your deployment (see [Pipeline Configuration](#pipeline-configuration))
4. **Push to GitLab**: Push your code to trigger the automated pipeline

## Pipeline Stages

The pipeline includes three stages:

1. **Build**: Builds your Docker image and pushes it to the container registry
2. **Deploy**: Deploys your application to GKE
   - Production deployments on `main` branch
   - Preview environments for merge requests
3. **Cleanup**: Manual cleanup of merge request preview environments

## Pipeline Configuration

The pipeline behavior can be controlled via environment variables. For a complete list of available configuration options and environment variables, refer to the [ci-library README](https://code.pan.run/airs/demo-env/ci-library/-/blob/main/README.md).

Key configuration points are marked in [.gitlab-ci.yml](.gitlab-ci.yml) with comments directing to the ci-library documentation.

### Using a Local Helm Chart

By default, this template uses the standard deployment job (`.gke-deploy-with-mr`). If you want to use a Helm chart stored within your repository instead of the default deployment method, you need to:

1. Change the job reference from `.gke-deploy-with-mr` to `.gke-deploy-local-chart-with-mr` in your [.gitlab-ci.yml](.gitlab-ci.yml)
2. Refer to the [ci-library documentation](https://code.pan.run/airs/demo-env/ci-library/-/blob/main/README.md) for complete configuration details and required variables for local chart deployments

## Google Cloud Service Accounts

If your application needs to access Google Cloud services (such as Vertex AI), you'll need to create service accounts with workload identity bindings. Use the [airs/demo-env/infra](https://code.pan.run/airs/demo-env/infra) repository to provision these resources.

For support with infrastructure setup, visit the `#airs-demo-env-infra` Slack channel.

## How It Works

The template uses shared CI/CD templates from the [airs/demo-env/ci-library](https://code.pan.run/airs/demo-env/ci-library) project, which provides:

- `.build`: Docker image building and registry push
- `.gke-deploy-with-mr`: Kubernetes deployment with merge request support
- `.gke-cleanup-mr`: Environment cleanup utilities

See [.gitlab-ci.yml](.gitlab-ci.yml) for the complete pipeline configuration.

<!---Protected_by_PANW_Code_Armor_2024 - Y3ByfC9haXJzL2RlbW8tZW52L3Byb2plY3QtdGVtcGxhdGVzL2s4cy1hcHAtdGVtcGxhdGV8MjQyMjZ8bWFpbg== --->
